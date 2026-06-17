from __future__ import annotations

import sys

from common_drtnet_runner import run_experiment


if __name__ == "__main__":
    run_experiment(
        "02_drt_single_scale_no_downsample",
        arch="DRTnet",
        single_scale_safa=True,
        extra_argv=sys.argv[1:],
    )
