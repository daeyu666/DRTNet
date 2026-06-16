from __future__ import annotations

import importlib
import sys

from common_drtnet_runner import DATA_ROOT, PROJECT_ROOT, _base_argv, _set_seed
from square_transformer_utils import patch_main_square_transformer


def run_square_transformer_experiment(extra_argv=None):
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

    sys.argv = [str(PROJECT_ROOT / "main.py")] + _base_argv("04_drt_square_transformer", "DRTnet")
    sys.argv += list(extra_argv or [])

    if "main" in sys.modules:
        del sys.modules["main"]
    main_mod = importlib.import_module("main")
    patch_main_square_transformer(main_mod)
    main_mod.main()


if __name__ == "__main__":
    run_square_transformer_experiment(sys.argv[1:])
