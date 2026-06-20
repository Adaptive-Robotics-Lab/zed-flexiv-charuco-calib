#!/usr/bin/env python3
"""Audit (or freely solve) the ZED 2i intrinsics from ChArUco views.

ChArUco recovers the full pinhole model (fx, fy, cx, cy + distortion). But the ZED left stream is
already rectified against a factory K the depth engine relies on -- so the DEFAULT is an AUDIT:
confirm the factory K reprojects sub-pixel and report a free solve for transparency, WITHOUT
overwriting K. Use --mode free only if you deliberately want an independent K.

    python scripts/calib_intrinsics.py --zed configs/zed_2i_hd720.yaml \
        --board configs/board_calibio_9x14.yaml --frames 20 --out zed2i_intrinsics.yaml
    # or, installed:  zfcc-intrinsics --zed ... --board ...
"""
from zfcc._cli import intrinsics_main

if __name__ == "__main__":
    intrinsics_main()
