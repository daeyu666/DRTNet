from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("DRTNET_ROOT", HERE.parent)).resolve()
DATA_ROOT = Path(os.environ.get("DRTNET_DATA_ROOT", PROJECT_ROOT / "data")).resolve()
RUN_ROOT = Path(os.environ.get("DRTNET_RUN_ROOT", PROJECT_ROOT / "runs" / "paviau_shadow_ablation")).resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loader import build_datasets
from metrics import calc_ergas, calc_psnr, calc_rmse, calc_sam, mrae
from models.srg_caun import build_srg_caun_hier_match


EXPERIMENT_NAME = "06_srg_caun"


def set_seed(seed: int = 10) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def to_unit_range(x: torch.Tensor) -> torch.Tensor:
    return x.float() / 255.0


def to_metric_range(x: torch.Tensor) -> np.ndarray:
    return (x.detach().cpu().numpy() * 255.0).clip(0.0, 255.0)


def unwrap_output(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    if isinstance(output, dict):
        for key in ("out", "pred", "prediction", "reconstruction"):
            if key in output and torch.is_tensor(output[key]):
                return output[key]
    raise TypeError("Unsupported SRG-CAUN output type: {}".format(type(output)))


def infer(model, test_list, device):
    test_ref, test_lr, test_hr = test_list
    ref = to_unit_range(test_ref).to(device)
    lr = to_unit_range(test_lr).to(device)
    hr = to_unit_range(test_hr).to(device)
    model.eval()
    with torch.no_grad():
        out = unwrap_output(model(lr, hr))
    return to_metric_range(ref), to_metric_range(out)


def validate(model, test_list, epoch, n_epochs, device):
    ref_np, out_np = infer(model, test_list, device)
    psnr = calc_psnr(ref_np, out_np)
    rmse = calc_rmse(ref_np, out_np)
    ergas = calc_ergas(ref_np, out_np)
    sam = calc_sam(ref_np, out_np)
    mrae_value = mrae(out_np, ref_np)
    print(
        "Val_Epoch_%d/%d, PSNR: %.4f, SAM: %.4f, RMSE: %.4f, ERGAS: %.4f, MRAE: %.4f"
        % (epoch, n_epochs, psnr, sam, rmse, ergas, mrae_value)
    )
    return psnr, {"psnr": psnr, "sam": sam, "rmse": rmse, "ergas": ergas, "mrae": mrae_value}


def train_one_epoch(model, train_list, optimizer, criterion, image_size, scale_ratio, epoch, n_epochs, device):
    train_ref, _train_lr, train_hr = train_list
    h, w = train_ref.size(2), train_ref.size(3)
    h_str = random.randint(0, h - image_size - 1)
    w_str = random.randint(0, w - image_size - 1)

    ref = train_ref[:, :, h_str:h_str + image_size, w_str:w_str + image_size]
    hr = train_hr[:, :, h_str:h_str + image_size, w_str:w_str + image_size]

    ref = to_unit_range(ref).to(device)
    hr = to_unit_range(hr).to(device)
    lr = F.interpolate(ref, scale_factor=1.0 / float(scale_ratio), mode="bicubic", align_corners=False)

    model.train()
    optimizer.zero_grad()
    out = unwrap_output(model(lr, hr))
    loss = criterion(out, ref)
    loss.backward()
    optimizer.step()

    print("Train_Epoch_%d/%d, Loss: %.6f" % (epoch, n_epochs, loss.item()))
    return loss.item()


def save_outputs(model, test_list, output_dir, device):
    ref_np, out_np = infer(model, test_list, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    scio.savemat(str(output_dir / "srg_caun_paviau_output.mat"), {
        "ref": np.squeeze(ref_np),
        "out": np.squeeze(out_np),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DATA_ROOT))
    parser.add_argument("--dataset", default="PaviaU")
    parser.add_argument("--scale_ratio", type=int, default=4)
    parser.add_argument("--n_select_bands", type=int, default=5)
    parser.add_argument("--n_bands", type=int, default=103)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--num_stages", type=int, default=3)
    parser.add_argument("--ref_topk", type=int, default=4)
    parser.add_argument("--ref_window", type=int, default=11)
    parser.add_argument("--ref_fine_window", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--save_outputs", action="store_true")
    args = parser.parse_args()

    if args.dataset != "PaviaU":
        raise ValueError("This script follows the DRT PaviaU setting; got dataset={}".format(args.dataset))

    data_root = Path(args.root)
    if not (data_root / "PaviaU.mat").exists():
        raise FileNotFoundError("Expected PaviaU.mat at {}".format(data_root / "PaviaU.mat"))

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = RUN_ROOT / EXPERIMENT_NAME
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path) if args.model_path else save_dir / "best_PaviaU_srg_caun.pkl"

    print(args)
    print("SRG-CAUN uses DRT's PaviaU setting: image_size=128, scale_ratio=4, n_select_bands=5, epochs=10000, lr=1e-4, seed=10.")
    print("Data scale: data_loader returns 0-255; this script trains SRG-CAUN in 0-1 and computes metrics after converting back to 0-255.")
    print("SRG-CAUN reference matching: ref_window={}, ref_fine_window={}, ref_topk={}.".format(args.ref_window, args.ref_fine_window, args.ref_topk))

    train_list, test_list = build_datasets(
        str(data_root),
        args.dataset,
        args.image_size,
        args.n_select_bands,
        args.scale_ratio,
    )

    model = build_srg_caun_hier_match(
        n_bands=args.n_bands,
        n_msi_bands=args.n_select_bands,
        scale_ratio=args.scale_ratio,
        hidden_dim=args.hidden_dim,
        num_stages=args.num_stages,
        ref_topk=args.ref_topk,
        ref_window=args.ref_window,
        ref_fine_window=args.ref_fine_window,
    ).to(device)

    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        print("Loaded checkpoint:", model_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    parameter_nums = sum(p.numel() for p in model.parameters())
    print("Model size:", str(float(parameter_nums / 1e6)) + "M")

    best_psnr, best_metrics = validate(model, test_list, 0, args.n_epochs, device)
    best_epoch = 0

    for epoch in range(args.n_epochs):
        train_one_epoch(
            model,
            train_list,
            optimizer,
            criterion,
            args.image_size,
            args.scale_ratio,
            epoch,
            args.n_epochs,
            device,
        )
        recent_psnr, recent_metrics = validate(model, test_list, epoch, args.n_epochs, device)
        if recent_psnr > best_psnr:
            best_psnr = recent_psnr
            best_metrics = recent_metrics
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
            print("Saved!", model_path)
        print("best psnr:", best_psnr, "at epoch:", best_epoch)

    print("best metrics:", best_metrics, "at epoch:", best_epoch)
    if args.save_outputs:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        save_outputs(model, test_list, save_dir / "evaluation", device)
        print("Saved outputs to", save_dir / "evaluation")


if __name__ == "__main__":
    main()
