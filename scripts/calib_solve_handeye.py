#!/usr/bin/env python3
"""Solve (offline) the eye-to-hand calibration from a captured session, run the full validation
suite (diversity gate, 5-solver cross-check, robot-world cross-check, AX=XB residuals, leave-one-out),
and write the drop-in T_base_zed2i.yaml + a JSON report. No hardware needed. Refuses to write on a
FAIL verdict unless --force.

    python scripts/calib_solve_handeye.py --session runs/session_001 \
        --board configs/board_calibio_9x14.yaml --out T_base_zed2i.yaml
    # or, installed:  zfcc-solve --session ... --out ...
"""
from zfcc._cli import solve_main

if __name__ == "__main__":
    solve_main()
