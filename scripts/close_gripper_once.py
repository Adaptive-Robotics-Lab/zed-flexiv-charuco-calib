#!/usr/bin/env python3
"""One-off: fully close the Flexiv-GraspG2 gripper (for the touch test probe tip).

Run:  /home/lab/miniconda3/envs/flexiv_env/bin/python scripts/close_gripper_once.py
Init() homes the fingers (they will move); keep hands clear.
"""
import time

import flexivrdk

robot = flexivrdk.Robot("Rizon4s-062626")
if robot.fault():
    robot.ClearFault()
robot.Enable()
g = flexivrdk.Gripper(robot)
g.Enable("Flexiv-GraspG2")
g.Init()
time.sleep(1.0)
g.Move(0.0, 0.02, 20.0)  # width 0 m, 0.02 m/s, 20 N — gentle full close
time.sleep(3.0)
try:
    st = g.states()
    print("gripper width now:", getattr(st, "width", "?"), "m")
except Exception as e:
    print("(state read skipped:", e, ")")
print("done — fingers closed")
