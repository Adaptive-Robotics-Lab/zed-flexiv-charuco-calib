#!/usr/bin/env python3
"""Physical touch test: the end-to-end accuracy check that actually predicts grasp success.

Given a solved T_base_zed2i.yaml, picks a target the camera sees (a ChArUco corner), transforms it
into the base frame, and prints the base-frame coordinate to command the TCP to. You move the TCP
there and measure the real miss with a ruler -- the metric that would have caught the old ~1.5-2cm
height error (a great reprojection RMS can still hide a bad chain).

    python scripts/run_touch_test.py --calib T_base_zed2i.yaml \
        --board configs/board_calibio_9x14.yaml --zed configs/zed_2i_hd720.yaml --corner 0
    # or, installed:  zfcc-touch-test --calib ...
"""
from zfcc._cli import touch_test_main

if __name__ == "__main__":
    touch_test_main()
