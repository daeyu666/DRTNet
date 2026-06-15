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
        "--root",
        str(DATA_ROOT),
        "--dataset",
        "PaviaU",
        "--arch",
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


def _select_tensor(items: list[torch.Tensor]) -> torch.Tensor:
    index = int(os.environ.get("DRTNET_SINGLE_SCALE_INDEX", "-1"))
    return items[index]


def _single_scale_value(value):
    if torch.is_tensor(value) and value.dim() >= 3:
        return value
    if isinstance(value, (list, tuple)):
        tensors = [item for item in value if torch.is_tensor(item) and item.dim() >= 3]
        if len(tensors) >= 2:
            selected = _select_tensor(tensors)
            return type(value)(selected if torch.is_tensor(item) and item.dim() >= 3 else item for item in value)
    return value


def _patch_module_forward_to_single_scale(module) -> None:
    if getattr(module, "_drtnet_single_scale_forward", False):
        return

    old_forward = module.forward

    def forward_single_scale(*args, **kwargs):
        tensor_args = [arg for arg in args if torch.is_tensor(arg) and arg.dim() >= 3]
        if len(tensor_args) >= 2:
            selected = _select_tensor(tensor_args)
            args = tuple(selected if torch.is_tensor(arg) and arg.dim() >= 3 else arg for arg in args)
        else:
            args = tuple(_single_scale_value(arg) for arg in args)
            kwargs = {key: _single_scale_value(value) for key, value in kwargs.items()}
        return old_forward(*args, **kwargs)

    module.forward = forward_single_scale
    module._drtnet_single_scale_forward = True


def _patch_safa_to_single_scale(main_mod) -> None:
    if not hasattr(main_mod, "MCT_rectangle"):
        raise RuntimeError("Cannot patch SAFA: main.MCT_rectangle is not available.")

    base_cls = main_mod.MCT_rectangle

    class SingleScaleSAFAMCT(base_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            patched = []
            for name, module in self.named_modules():
                signature = "{} {}".format(name, module.__class__.__name__).lower()
                if "safa" in signature or "scale" in signature and "aggregation" in signature:
                    module.single_scale = True
                    module.use_multiscale = False
                    module.safa_mode = "single_scale"
                    _patch_module_forward_to_single_scale(module)
                    patched.append(name or module.__class__.__name__)
            self.drtnet_single_scale_safa_modules = patched
            print("Single-scale SAFA patch modules:", patched if patched else "none found")

    main_mod.MCT_rectangle = SingleScaleSAFAMCT
    os.environ["DRTNET_SAFA_MODE"] = "single_scale"
    os.environ["DRTNET_SINGLE_SCALE_SAFA"] = "1"


def run_experiment(
    experiment_name: str,
    arch: str = "MCT_rectangle",
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
