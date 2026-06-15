from __future__ import annotations

import importlib
import os
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


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
        return main_mod.train(
            train_list,
            image_size,
            scale_ratio,
            n_bands,
            arch,
            model,
            optimizer,
            criterion,
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


def _patch_multiscale_pooling_to_single_scale(module) -> None:
    if getattr(module, "_drtnet_single_scale_forward", False):
        return
    required = [
        "conv3", "conv2d", "PWconv_pool_in", "PWconv_pool_out",
        "ca", "sa", "context"
    ]
    if not all(hasattr(module, name) for name in required):
        return

    def forward_single_scale(x):
        x_res = x
        data_3d = module.conv3(x)
        x = module.conv2d(x)
        x = data_3d + x
        b, c, h, w = x.shape

        x_cbam = x
        x = module.PWconv_pool_in(x)
        x_in = x.reshape(b, c, h, w)

        # Current-scale ablation: replace pooled [1, 2, 4, 8] aggregation with
        # the current-resolution feature only.
        x_out = module.PWconv_pool_out(x_in)
        x_out = x_res + x_out
        x_cbam = x_cbam * module.ca(x_cbam)
        x_cbam = x_cbam * module.sa(x_cbam)
        x_weight = module.context(x_out)
        x_out = x_out + x_in + x_cbam
        x_out = x_weight * x_out
        return x_out

    module.forward = forward_single_scale
    module.single_scale = True
    module.use_multiscale = False
    module.safa_mode = "single_scale"
    module._drtnet_single_scale_forward = True


def _patch_safa_to_single_scale(main_mod) -> None:
    candidate_names = ["DRTnet", "MCT_rectangle", "MCT"]
    base_cls = None
    selected_name = None
    for name in candidate_names:
        if hasattr(main_mod, name):
            base_cls = getattr(main_mod, name)
            selected_name = name
            break
    if base_cls is None:
        print("No DRT/MCT model class found; skipping single-scale SAFA patch.")
        return

    class SingleScaleDRT(base_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            patched = []
            for module_name, module in self.named_modules():
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
                    patched.append(module_name or module.__class__.__name__)
            self.drtnet_single_scale_safa_modules = patched
            if patched:
                print("Single-scale SAFA patch modules:", patched)
            else:
                print("No SAFA/multiscale pooling module found; single-scale ablation is not active.")

    setattr(main_mod, selected_name, SingleScaleDRT)
    os.environ["DRTNET_SAFA_MODE"] = "single_scale"
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
