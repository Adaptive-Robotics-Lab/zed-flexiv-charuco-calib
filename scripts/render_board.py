#!/usr/bin/env python3
"""Render the configured ChArUco board to a PNG so you can verify it matches the physical Calib.io
target (square/marker counts, dictionary, parity) BEFORE capturing -- a mismatch here is the #1
silent cause of a confidently-wrong calibration.

    python scripts/render_board.py --board configs/board_calibio_9x14.yaml --out board.png
    # or, installed:  zfcc-render-board --board ... --out board.png
"""
from zfcc._cli import render_board_main

if __name__ == "__main__":
    render_board_main()
