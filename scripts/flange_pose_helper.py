#!/usr/bin/env python3
"""Persistent Flexiv flange-pose reader for collect_web.py (JSON lines over stdio).

Runs in an env that has ``flexivrdk`` (e.g. flexiv_env) while the camera process runs
elsewhere. Mirrors zfcc.robot_io.FlangePoseReader's rdk transport exactly: connect once,
ClearFault if needed, Enable, then answer each "get" line with the current flange pose7
``[x,y,z,qw,qx,qy,qz]``. READ-ONLY: no motion commands are ever sent.

Usage: flange_pose_helper.py <robot_serial>
Protocol (every protocol line is prefixed "ZFCC " because flexivrdk logs to stdout):
  stdin "get\n" -> stdout 'ZFCC {"pose7": [...]}'   |   stdin "quit\n" -> exit
"""
from __future__ import annotations

import json
import sys


def emit(obj) -> None:
    print("ZFCC " + json.dumps(obj), flush=True)


def main() -> None:
    target = sys.argv[1]
    try:
        import flexivrdk
        robot = flexivrdk.Robot(target)
        if robot.fault():
            robot.ClearFault()
        robot.Enable()
    except Exception as e:
        emit({"ready": False, "error": str(e)})
        return
    emit({"ready": True})

    for line in sys.stdin:
        cmd = line.strip()
        if cmd == "quit":
            break
        if cmd != "get":
            continue
        try:
            st = robot.states()
            pose = getattr(st, "flange_pose", None)
            if pose is None:
                pose = st.tcp_pose  # requires tool==flange (documented in PROCEDURE)
            emit({"pose7": [float(v) for v in pose]})
        except Exception as e:
            emit({"error": str(e)})


if __name__ == "__main__":
    main()
