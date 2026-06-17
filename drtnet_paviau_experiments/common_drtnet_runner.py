from __future__ import annotations

import importlib
import os
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("DRTNET_ROOT", HERE.parent)).resolve()
DATA_ROOT = Path(os.environ.get("DRTNET_DATA_ROOT", PROJECT_ROOT / "data")).resolve()
RUN_ROOT = Path(os.environ.get("DRTNET_RUN_ROOT", PROJECT_ROOT / "runs" / "paviau_shadow_ablation")).resolve()
DEFAULT_ARCH = os.environ.get("DRTNET_ARCH", "DRTnet")


def _set_seed(seed: int = 10) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def _base_argv(experiment_name: str, arch: str) -> list[str]:
    save_dir = RUN_ROOT / experiment_name
    model_path = save_dir / "best_PaviaU_{}.pkl".format(arch)
    save_dir.mkdir(parents=True, exist_ok=True)

    return [
        "-root",
        str(DATA_ROOT),
        "-dataset",
        "PaviaU",
        "-arch",
        arch,
        "--image_size",
        "128",
        "--n_select_bands",
        "5",
        "--scale_ratio",
        "4",
        "--n_epochs",
        "10000",
        "--lr",
        "0.0001",
        "--model_path",
        str(model_path),
    ]


def _patch_no_contrast(main_mod) -> None:
    if not hasattr(main_mod, "train_contrast"):
        print("No contrastive training function found; this code path is already non-contrastive.")
        return
    if not hasattr(main_mod, "train"):
        raise RuntimeError("Cannot disable contrastive learning: main.train is not available.")

    def train_without_contrast(
        train_list,
        image_size,
        scale_ratio,
        n_bands,
        arch,
        model,
        _encoder,
        optimizer,
        criterion,
        _contrast_loss,
        epoch,
        n_epochs,
    ):
        l1 = torch.nn.L1Loss()
        if torch.cuda.is_available():
            l1 = l1.cuda()
        return main_mod.train(
            train_list,
            image_size,
            scale_ratio,
            n_bands,
            arch,
            model,
            optimizer,
            criterion,
            l1,
            epoch,
            n_epochs,
        )

    class NoOpMoco:
        def __init__(self, *args, **kwargs):
            pass

        def cuda(self):
            return self

    main_mod.train_contrast = train_without_contrast
    main_mod.Moco = NoOpMoco


def _patch_safa_to_single_scale(main_mod) -> None:
    if not hasattr(main_mod, "MCT_rectangle"):
        print("No MCT_rectangle model class found; skipping single-scale DRT patch.")
        return

    base_cls = main_mod.MCT_rectangle
    from models.transformer_rectangle import TransformerModel_rectangle

    class SingleScaleDRT(base_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            u1_channel = self.multi_scale_fusion.len

            self.single_scale_b = nn.Sequential(
                nn.Conv2d(self.n_bands, self.n_bands * 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.n_bands * 2, self.n_bands * 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.single_scale_c = nn.Sequential(
                nn.Conv2d(self.n_bands, self.n_bands * 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.n_bands * 2, self.n_bands * 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.single_scale_transformer1 = TransformerModel_rectangle(
                map_size=32,
                M_channel=self.n_bands * 2,
                dim=128,
                depth=5,
                heads=8,
                mlp_dim=self.n_bands,
                dropout_rate=0.1,
                attn_dropout_rate=0.1,
            )
            self.single_scale_f1_reduce = nn.Sequential(
                nn.Conv2d(self.n_bands * 6, u1_channel, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
            )
            self.drtnet_single_scale_safa_modules = [
                "single_scale_b",
                "single_scale_c",
                "single_scale_transformer1",
                "single_scale_f1_reduce",
                "multi_scale_fusion",
                "multi_scale_fusion2",
            ]
            print("Single-scale DRT patch active: D1/D2 downsample branches are bypassed; SPP modules are kept.")

        def _attend_n_band(self, x):
            x = x * self.ca1(x)
            x = x * self.sa(x)
            return x

        def _attend_2n_band(self, x):
            x = x * self.ca(x)
            x = x * self.sa(x)
            return x

        def forward(self, x_lr, x_hr):
            if self.arch not in ("DRTnet", "DRTnet_GSIS", "no_contrast"):
                return super().forward(x_lr, x_hr)

            current_size = x_lr.shape[-2:]
            if current_size != (32, 32):
                raise ValueError(
                    "Single-scale DRT ablation expects 32x32 LR features. "
                    "Use --image_size 128 with --scale_ratio 4 to match the paper setting."
                )

            x_hr_current = F.interpolate(x_hr, size=current_size, mode="bilinear")
            a = self.conv1(x_hr_current)
            a = self._attend_n_band(a)

            x_lr_current = self._attend_n_band(x_lr)
            b = self.single_scale_b(a)
            b = self._attend_2n_band(b)
            c = self.single_scale_c(x_lr_current)
            c = self._attend_2n_band(c)

            d = F.interpolate(x_lr, scale_factor=self.scale_ratio, mode="bilinear")
            d = self._attend_n_band(d)

            transformer_results = self.single_scale_transformer1(b, c)
            e = transformer_results["z"]
            f1 = torch.cat((torch.cat((b, c), 1), e), 1)
            f1 = self.single_scale_f1_reduce(f1)
            f1 = self.multi_scale_fusion(f1)
            f1 = self.U1(f1)

            transformer_results1 = self.transformer2(a, x_lr)
            g = transformer_results1["z"]
            f2 = torch.cat((torch.cat((a, x_lr), 1), g), 1)
            f2 = torch.cat((f2, f1), 1)
            f2 = self.MS_MSA2(f2, self.n_bands * 4)
            f2 = self.convf2(f2)
            f2 = self.deconv2(f1, f2)
            f2 = self.multi_scale_fusion2(f2)
            f2 = F.interpolate(f2, scale_factor=self.scale_ratio, mode="bilinear")
            f2 = self.U2(f2)

            x = torch.cat((f2, x_hr), 1)
            x = torch.cat((x, d), 1)
            x = self.conv3(x)
            x_spat = x + self.conv_spat(x)
            x_spec = x_spat + self.conv_spec(x_spat)
            return x_spec, x_spec

    main_mod.MCT_rectangle = SingleScaleDRT
    os.environ["DRTNET_SAFA_MODE"] = "single_scale_current_resolution"
    os.environ["DRTNET_SINGLE_SCALE_SAFA"] = "1"


def run_experiment(
    experiment_name: str,
    arch: str = DEFAULT_ARCH,
    *,
    disable_contrast: bool = False,
    single_scale_safa: bool = False,
    extra_argv: Iterable[str] | None = None,
) -> None:
    if not (DATA_ROOT / "PaviaU.mat").exists():
        raise FileNotFoundError(
            "Expected PaviaU.mat at {}. Set DRTNET_DATA_ROOT if your data folder is elsewhere.".format(DATA_ROOT)
        )
    if not (PROJECT_ROOT / "main.py").exists():
        raise FileNotFoundError(
            "Expected DRTNet main.py at {}. Set DRTNET_ROOT to the DRTNet project root.".format(PROJECT_ROOT)
        )

    _set_seed(10)
    sys.path.insert(0, str(PROJECT_ROOT))
    script_overrides = list(extra_argv or [])
    sys.argv = [str(PROJECT_ROOT / "main.py")] + _base_argv(experiment_name, arch) + script_overrides

    main_mod = importlib.import_module("main")

    if single_scale_safa:
        _patch_safa_to_single_scale(main_mod)
    if disable_contrast:
        _patch_no_contrast(main_mod)

    main_mod.main()
