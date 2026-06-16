from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

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
from drtnet_paviau_experiments.common_drtnet_runner import _patch_multiscale_pooling_to_single_scale
from drtnet_paviau_experiments.square_transformer_utils import replace_with_square_window_transformers


EXPERIMENT_NAME = "05_drt_plain_baseline"
DESCRIPTION = "Plain DRT baseline: no CESR, square-window transformer, single-scale SAFA, fused-output MSE only"


def apply_single_scale_safa(model):
    patched = []
    for module_name, module in model.named_modules():
        signature = "{} {}".format(module_name, module.__class__.__name__).lower()
        if (
            "safa" in signature
            or "multiscale" in signature
            or "multi_scale" in signature
            or ("scale" in signature and "pool" in signature)
            or ("scale" in signature and "fusion" in signature)
            or ("scale" in signature and "aggregation" in signature)
        ):
            _patch_multiscale_pooling_to_single_scale(module)
            if getattr(module, "_drtnet_single_scale_forward", False):
                patched.append(module_name or module.__class__.__name__)
    print("Single-scale SAFA modules:", patched)


def build_plain_baseline_model():
    model = MCT_rectangle("DRTnet", 4, 5, 103, "PaviaU")
    replace_with_square_window_transformers(model)
    apply_single_scale_safa(model)
    print("Plain baseline evaluation active: no CESR, square-window transformer, single-scale SAFA.")
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def sam_map(ref_chw, out_chw, eps=1e-8):
    numerator = np.sum(ref_chw * out_chw, axis=0)
    denominator = np.linalg.norm(ref_chw, axis=0) * np.linalg.norm(out_chw, axis=0) + eps
    cosine = np.clip(numerator / denominator, -1.0, 1.0)
    return np.arccos(cosine) * 180.0 / np.pi


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


def auto_shadow_pixels(ref_chw, count):
    brightness = ref_chw.mean(axis=0)
    order = np.argsort(brightness.reshape(-1))[:count]
    width = brightness.shape[1]
    return [(int(idx // width), int(idx % width)) for idx in order]


def center_crop_offset(root, image_size, scale_ratio):
    img = scio.loadmat(str(Path(root) / "PaviaU.mat"))["paviaU"] * 1.0
    h_edge = img.shape[1] // scale_ratio * scale_ratio - img.shape[1]
    w_edge = img.shape[0] // scale_ratio * scale_ratio - img.shape[0]
    h_edge = -1 if h_edge == 0 else int(h_edge)
    w_edge = -1 if w_edge == 0 else int(w_edge)
    img = img[:w_edge, :h_edge, :]
    width, height, _ = img.shape
    return (width - image_size) // 2, (height - image_size) // 2


def save_curve_plot(output_dir, pixel, ref_chw, out_chw):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row, col = pixel
    bands = np.arange(ref_chw.shape[0])
    plt.figure(figsize=(8, 4.5))
    plt.plot(bands, ref_chw[:, row, col], label="GT", linewidth=2.0)
    plt.plot(bands, out_chw[:, row, col], label=EXPERIMENT_NAME, linewidth=1.5)
    plt.xlabel("Band")
    plt.ylabel("Normalized intensity")
    plt.title("Spectral curve at test pixel ({}, {})".format(row, col))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "{}_spectral_curve_r{}_c{}.png".format(EXPERIMENT_NAME, row, col), dpi=180)
    plt.close()


def save_pixel_spectra_csv(output_dir, pixels, ref_chw, out_chw, offset):
    with (output_dir / "{}_shadow_pixel_spectra.csv".format(EXPERIMENT_NAME)).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test_row", "test_col", "source_row", "source_col", "band", "GT", EXPERIMENT_NAME])
        for row, col in pixels:
            source_row = offset[0] + row
            source_col = offset[1] + col
            for band in range(ref_chw.shape[0]):
                writer.writerow([row, col, source_row, source_col, band, ref_chw[band, row, col], out_chw[band, row, col]])


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
    output_dir = Path(args.output_dir) if args.output_dir else run_root / "evaluation_plain_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = run_root / EXPERIMENT_NAME / "best_PaviaU_DRTnet.pkl"
    if not checkpoint.exists():
        raise FileNotFoundError("Missing checkpoint: {}".format(checkpoint))

    _, test_list = build_datasets(args.root, "PaviaU", args.image_size, 5, args.scale_ratio)
    test_ref, test_lr, test_hr = test_list
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_plain_baseline_model()
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    ref = test_ref.to(device).float()
    lr = test_lr.to(device).float()
    hr = test_hr.to(device).float()
    with torch.no_grad():
        out, _ = model(lr, hr)

    ref_np = ref.detach().cpu().numpy()
    out_np = out.detach().cpu().numpy()
    row = {
        "experiment": EXPERIMENT_NAME,
        "description": DESCRIPTION,
        "rmse": float(calc_rmse(ref_np, out_np)),
        "psnr": float(calc_psnr(ref_np, out_np)),
        "ergas": float(calc_ergas(ref_np, out_np)),
        "sam": float(calc_sam(ref_np, out_np)),
        "mrae": float(mrae(out_np, ref_np)),
    }

    metric_path = output_dir / "metrics.csv"
    with metric_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "description", "rmse", "psnr", "ergas", "sam", "mrae"])
        writer.writeheader()
        writer.writerow(row)

    ref_chw = np.squeeze(ref_np)
    out_chw = np.squeeze(out_np)
    scio.savemat(str(output_dir / "{}.mat".format(EXPERIMENT_NAME)), {"ref": ref_chw, "out": out_chw})
    scio.savemat(str(output_dir / "{}_sam_map.mat".format(EXPERIMENT_NAME)), {"sam_map": sam_map(ref_chw, out_chw)})

    pixels = parse_pixels(args.pixels)
    if not pixels:
        pixels = auto_shadow_pixels(ref_chw, args.auto_shadow_pixels)
    offset = center_crop_offset(args.root, args.image_size, args.scale_ratio)
    save_pixel_spectra_csv(output_dir, pixels, ref_chw, out_chw, offset)
    for pixel in pixels:
        save_curve_plot(output_dir, pixel, ref_chw, out_chw)

    print("Saved metrics:", metric_path)
    print("{experiment}: PSNR={psnr:.4f}, SAM={sam:.4f}, RMSE={rmse:.4f}, ERGAS={ergas:.4f}, MRAE={mrae:.4f}".format(**row))
    print("Saved spectral curves for test pixels:", pixels)
    print("Test patch source offset in PaviaU:", offset)


if __name__ == "__main__":
    main()
