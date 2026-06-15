from __future__ import annotations

import sys

from common_drtnet_runner import run_experiment


if __name__ == "__main__":
    run_experiment(
        "01_drt_normal",
        arch="MCT",
        extra_argv=sys.argv[1:],
    )
