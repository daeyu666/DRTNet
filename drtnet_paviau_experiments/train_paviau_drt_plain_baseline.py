from __future__ import annotations

import importlib
import sys

from common_drtnet_runner import (
    DATA_ROOT,
    PROJECT_ROOT,
    _base_argv,
    _patch_no_contrast,
    _patch_safa_to_single_scale,
    _set_seed,
)
from square_transformer_utils import patch_main_square_transformer


EXPERIMENT_NAME = "05_drt_plain_baseline"


def run_plain_baseline_experiment(extra_argv=None):
    """Train the minimal DRT ablation on PaviaU.

    Baseline definition:
    - no CESR contrastive loss;
    - rectangle transformer replaced by ordinary square-window transformer;
    - SAFA multi-scale aggregation replaced by current-scale aggregation;
    - training loss is the final fused-output MSE only.
    """
    if not (DATA_ROOT / "PaviaU.mat").exists():
        raise FileNotFoundError(
            "Expected PaviaU.mat at {}. Set DRTNET_DATA_ROOT if your data folder is elsewhere.".format(DATA_ROOT)
        )
    if not (PROJECT_ROOT / "main.py").exists():
        raise FileNotFoundError(
            "Expected DRTNet main.py at {}. Set DRTNET_ROOT to the DRTNet project root.".format(PROJECT_ROOT)
        )

    _set_seed(10)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    sys.argv = [str(PROJECT_ROOT / "main.py")] + _base_argv(EXPERIMENT_NAME, "DRTnet")
    sys.argv += list(extra_argv or [])

    if "main" in sys.modules:
        del sys.modules["main"]
    main_mod = importlib.import_module("main")

    patch_main_square_transformer(main_mod)
    _patch_safa_to_single_scale(main_mod)
    _patch_no_contrast(main_mod)

    print("Plain baseline ablation active: no CESR, square-window transformer, single-scale SAFA, fused-output MSE only.")
    main_mod.main()


if __name__ == "__main__":
    run_plain_baseline_experiment(sys.argv[1:])
