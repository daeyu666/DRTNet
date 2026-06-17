from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io as scio
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("DRTNET_ROOT", HERE.parent)).resolve()
DATA_ROOT = Path(os.environ.get("DRTNET_DATA_ROOT", PROJECT_ROOT / "data")).resolve()
RUN_ROOT = Path(os.environ.get("DRTNET_RUN_ROOT", PROJECT_ROOT / "runs" / "paviau_shadow_ablation")).resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loader import build_datasets
from metrics import calc_ergas, calc_psnr, calc_rmse, calc_sam, mrae
from models.MCT_rectangle import MCT_rectangle
from drtnet_paviau_experiments.common_drtnet_runner import _patch_safa_to_single_scale
from drtnet_paviau_experiments.square_transformer_utils import replace_with_square_window_transformers


EXPERIMENTS = [
    ("01_drt_normal", "DRT + rectangle transformer + SAFA + CESR", "rectangle"),
    (
        "02_drt_single_scale_no_downsample",
        "DRT + rectangle transformer + current-resolution single-scale branch + CESR",
        "single_scale_safa",
    ),
    ("03_drt_no_contrast", "DRT + rectangle transformer + SAFA, no CESR", "rectangle"),
    ("04_drt_square_transformer", "DRT + square-window transformer + SAFA + CESR", "square_transformer"),
]

CHECKPOINT_ALIASES = {
    "02_drt_single_scale_no_downsample": ["02_drt_safa_single_scale", "02_drt_single_scale_safa"],
}


def parse_pixels(values):
    pixels = []
    for value in values or []:
        for item in value.split(";"):
            item = item.strip()
            if not item:
                continue
            row, col = item.split(",")
            pixels.append((int(row), int(col)))
    return pixels


def center_crop_offset(root, image_size, scale_ratio):
    img = scio.loadmat(str(Path(root) / "PaviaU.mat"))["paviaU"] * 1.0
    h_edge = img.shape[1] // scale_ratio * scale_ratio - img.shape[1]
    w_edge = img.shape[0] // scale_ratio * scale_ratio - img.shape[0]
    h_edge = -1 if h_edge == 0 else int(h_edge)
    w_edge = -1 if w_edge == 0 else int(w_edge)
    img = img[:w_edge, :h_edge, :]
    width, height, _ = img.shape
    return (width - image_size) // 2, (height - image_size) // 2


def build_single_scale_safa_model():
    namespace = SimpleNamespace(MCT_rectangle=MCT_rectangle)
    _patch_safa_to_single_scale(namespace)
    return namespace.MCT_rectangle("DRTnet", 4, 5, 103, "PaviaU")


def build_model(mode):
    if mode == "single_scale_safa":
        model = build_single_scale_safa_model()
    else:
        model = MCT_rectangle("DRTnet", 4, 5, 103, "PaviaU")

    if mode == "square_transformer":
        replace_with_square_window_transformers(model)
        print("Square-window transformer evaluation active: transformer1 8x8, transformer2 12x12")
    elif mode not in ("rectangle", "single_scale_safa"):
        raise ValueError("Unknown model mode: {}".format(mode))

    if torch.cuda.is_available():
        model = model.cuda()
    return model


def find_checkpoint(run_root, exp_name):
    names = [exp_name] + CHECKPOINT_ALIASES.get(exp_name, [])
    for name in names:
        checkpoint = run_root / name / "best_PaviaU_DRTnet.pkl"
        if checkpoint.exists():
            if name != exp_name:
                print("Using legacy checkpoint directory for {}: {}".format(exp_name, name))
            return checkpoint
    return run_root / exp_name / "best_PaviaU_DRTnet.pkl"


def load_checkpoint(model, checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def infer(model, test_list):
    test_ref, test_lr, test_hr = test_list
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref = test_ref.to(device).float()
    lr = test_lr.to(device).float()
    hr = test_hr.to(device).float()
    with torch.no_grad():
        out, _ = model(lr, hr)
    return ref.detach().cpu().numpy(), out.detach().cpu().numpy()


def metric_row(name, description, ref, out):
    return {
        "experiment": name,
        "description": description,
        "rmse": float(calc_rmse(ref, out)),
        "psnr": float(calc_psnr(ref, out)),
        "ergas": float(calc_ergas(ref, out)),
        "sam": float(calc_sam(ref, out)),
        "mrae": float(mrae(out, ref)),
    }


def sam_map(ref_chw, out_chw, eps=1e-8):
    numerator = np.sum(ref_chw * out_chw, axis=0)
    denominator = np.linalg.norm(ref_chw, axis=0) * np.linalg.norm(out_chw, axis=0) + eps
    cosine = np.clip(numerator / denominator, -1.0, 1.0)
    return np.arccos(cosine) * 180.0 / np.pi


def auto_shadow_pixels(ref, count):
    ref_chw = np.squeeze(ref)
    brightness = ref_chw.mean(axis=0)
    order = np.argsort(brightness.reshape(-1))[:count]
    width = brightness.shape[1]
    return [(int(idx // width), int(idx % width)) for idx in order]


def save_curve_plot(output_dir, pixel, ref, outputs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row, col = pixel
    bands = np.arange(ref.shape[0])
    plt.figure(figsize=(8, 4.5))
    plt.plot(bands, ref[:, row, col], label="GT", linewidth=2.0)
    for name, out in outputs.items():
        plt.plot(bands, out[:, row, col], label=name, linewidth=1.5)
    plt.xlabel("Band")
    plt.ylabel("Normalized intensity")
    plt.title("Spectral curve at test pixel ({}, {})".format(row, col))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "spectral_curve_r{}_c{}.png".format(row, col), dpi=180)
    plt.close()


def save_pixel_spectra_csv(output_dir, pixels, ref, outputs, offset):
    with (output_dir / "shadow_pixel_spectra.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test_row", "test_col", "source_row", "source_col", "band", "GT"] + list(outputs.keys()))
        for row, col in pixels:
            source_row = offset[0] + row
            source_col = offset[1] + col
            for band in range(ref.shape[0]):
                writer.writerow(
                    [row, col, source_row, source_col, band, ref[band, row, col]]
                    + [out[band, row, col] for out in outputs.values()]
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DATA_ROOT))
    parser.add_argument("--run_root", default=str(RUN_ROOT))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--pixels", action="append", help="Shadow pixels in test patch coords, e.g. --pixels 60,70;42,88")
    parser.add_argument("--auto_shadow_pixels", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--scale_ratio", type=int, default=4)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir) if args.output_dir else run_root / "evaluation_with_square"
    output_dir.mkdir(parents=True, exist_ok=True)

    _, test_list = build_datasets(args.root, "PaviaU", args.image_size, 5, args.scale_ratio)

    rows = []
    outputs = {}
    ref_chw = None
    for exp_name, description, mode in EXPERIMENTS:
        checkpoint = find_checkpoint(run_root, exp_name)
        if not checkpoint.exists():
            print("Skip missing checkpoint:", checkpoint)
            continue

        print("Evaluating", exp_name, checkpoint)
        model = build_model(mode)
        model = load_checkpoint(model, checkpoint)
        ref, out = infer(model, test_list)
        row = metric_row(exp_name, description, ref, out)
        rows.append(row)

        ref_chw = np.squeeze(ref)
        out_chw = np.squeeze(out)
        outputs[exp_name] = out_chw
        scio.savemat(str(output_dir / "{}.mat".format(exp_name)), {"ref": ref_chw, "out": out_chw})
        scio.savemat(str(output_dir / "{}_sam_map.mat".format(exp_name)), {"sam_map": sam_map(ref_chw, out_chw)})

    if rows:
        metric_path = output_dir / "metrics.csv"
        with metric_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["experiment", "description", "rmse", "psnr", "ergas", "sam", "mrae"])
            writer.writeheader()
            writer.writerows(rows)
        print("Saved metrics:", metric_path)
        for row in rows:
            print(
                "{experiment}: PSNR={psnr:.4f}, SAM={sam:.4f}, RMSE={rmse:.4f}, ERGAS={ergas:.4f}, MRAE={mrae:.4f}".format(
                    **row
                )
            )

    if ref_chw is not None and outputs:
        pixels = parse_pixels(args.pixels)
        if not pixels:
            pixels = auto_shadow_pixels(ref_chw[np.newaxis, ...], args.auto_shadow_pixels)
        offset = center_crop_offset(args.root, args.image_size, args.scale_ratio)
        save_pixel_spectra_csv(output_dir, pixels, ref_chw, outputs, offset)
        for pixel in pixels:
            save_curve_plot(output_dir, pixel, ref_chw, outputs)
        print("Saved spectral curves for test pixels:", pixels)
        print("Test patch source offset in PaviaU:", offset)


if __name__ == "__main__":
    main()
