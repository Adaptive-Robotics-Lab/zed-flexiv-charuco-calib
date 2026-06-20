#!/usr/bin/env python3
"""Capture an eye-to-hand session: for each robot pose, grab the ZED left frame, detect the ChArUco
board, solve its camera-frame pose, read the FLANGE pose, and append a Capture. Saves a session
directory you then solve offline with calib_solve_handeye.py.

Modes: --mode manual (hand-guide, ENTER to record) or --mode auto (step robot.joint_targets).
Needs the ZED SDK + a robot transport; the solve path does not.

    python scripts/calib_collect_handeye.py --session runs/session_001 \
        --board configs/board_calibio_9x14.yaml --zed configs/zed_2i_hd720.yaml \
        --robot configs/rizon4s.yaml --mode manual
    # or, installed:  zfcc-collect --session ... --mode manual
"""
from zfcc._cli import collect_main

if __name__ == "__main__":
    collect_main()
