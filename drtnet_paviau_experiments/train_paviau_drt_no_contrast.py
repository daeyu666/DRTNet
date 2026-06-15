from __future__ import annotations

import sys

from common_drtnet_runner import run_experiment


if __name__ == "__main__":
    run_experiment(
        "03_drt_no_contrast",
        arch="DRTnet",
        disable_contrast=True,
        extra_argv=sys.argv[1:],
    )
