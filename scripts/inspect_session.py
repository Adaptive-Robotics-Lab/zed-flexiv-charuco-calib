#!/usr/bin/env python3
"""Inspect a captured session WITHOUT solving: per-capture corner counts, focus, PnP RMS, and the
pose-diversity verdict. Run during/after capture to confirm you have enough diverse poses before you
walk away from the robot.

    python scripts/inspect_session.py --session runs/session_001
    # or, installed:  zfcc-inspect --session ...
"""
from zfcc._cli import inspect_main

if __name__ == "__main__":
    inspect_main()
