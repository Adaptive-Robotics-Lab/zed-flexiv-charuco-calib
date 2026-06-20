"""Eye-to-hand hand-eye calibration: recover the fixed ``T_base_camera`` for a FIXED camera
observing a ChArUco board mounted on the robot FLANGE.

THE one bookkeeping rule (eye-to-hand vs eye-in-hand):
  ``cv2.calibrateHandEye`` natively solves the eye-IN-hand problem AX=XB with X = T_cam_gripper,
  consuming (R/t)_gripper2base and (R/t)_target2cam. For a FIXED camera (eye-to-hand) the camera is
  not on the gripper; the board is. Feeding the INVERTED robot poses -- i.e. T_gripper2base :=
  invert(T_base_flange) -- makes the solver return X = T_base_camera directly. This module is the
  only place that inversion happens; ``test_handeye_synthetic`` asserts the sign explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import se3

__all__ = ["HE_METHODS", "RW_METHODS", "EyeToHandResult", "solve_eye_to_hand",
           "solve_robot_world", "cross_solver_spread", "axxb_residuals"]


def _cv():
    import cv2
    return cv2


# method name -> cv2 flag (resolved lazily so importing this module needs no cv2)
HE_METHODS = ("TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS")
RW_METHODS = ("SHAH", "LI")


@dataclass
class EyeToHandResult:
    T_base_camera: dict          # {method: 4x4}
    primary: str                 # "DANIILIDIS"
    T_primary: np.ndarray        # the primary 4x4
    n_poses: int


def solve_eye_to_hand(flange_poses_T, board_poses_T, methods=HE_METHODS) -> EyeToHandResult:
    """flange_poses_T: list of T_base_flange (4x4); board_poses_T: list of T_cam_board (4x4).

    Returns T_base_camera per method. DANIILIDIS is the primary (most rotation-robust)."""
    cv2 = _cv()
    if len(flange_poses_T) != len(board_poses_T) or len(flange_poses_T) < 3:
        raise ValueError("need >=3 paired poses for hand-eye")
    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    for T_bf, T_cb in zip(flange_poses_T, board_poses_T):
        T_g2b = se3.invert_T(T_bf)            # <-- the eye-to-hand inversion
        Rg, tg = se3.T_to_Rt(T_g2b)
        Rc, tc = se3.T_to_Rt(T_cb)            # target2cam, as-is
        R_g2b.append(Rg)
        t_g2b.append(tg.reshape(3, 1))
        R_t2c.append(Rc)
        t_t2c.append(tc.reshape(3, 1))
    out = {}
    for m in methods:
        flag = getattr(cv2, f"CALIB_HAND_EYE_{m}")
        R, t = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=flag)
        out[m] = se3.Rt_to_T(R, t)            # X = T_base_camera
    primary = "DANIILIDIS" if "DANIILIDIS" in out else list(out)[0]
    return EyeToHandResult(T_base_camera=out, primary=primary,
                           T_primary=out[primary], n_poses=len(flange_poses_T))


def solve_robot_world(flange_poses_T, board_poses_T, method="SHAH"):
    """Independent cross-check via cv2.calibrateRobotWorldHandEye (solves the AX=ZB system at once).

    Returns (T_base_camera, T_flange_board).

    Bookkeeping (derived by matching OpenCV's model equation to the eye-to-hand constraint):
      OpenCV solves   T_cam_world = T_cam_gripper @ T_gripper_base @ T_base_world.
      Our constraint  T_cam_board = T_camera_base @ T_base_flange  @ T_flange_board.
      Term-by-term => world=board, gripper=base(!), so:
        * input  world2cam   (R/t_world2cam)   = T_cam_board     (as-is)
        * input  base2gripper(R/t_base2gripper)= T_base_flange   (NOT inverted -- maps to T_gripper_base)
        * output gripper2cam (R/t_gripper2cam) = T_camera_base   -> T_base_camera = invert(gripper2cam)
        * output base2world  (R/t_base2world)  = T_board_flange  -> T_flange_board = invert(base2world)
    Verified to recover a known synthetic extrinsic to machine precision in test_handeye_synthetic.
    """
    cv2 = _cv()
    R_w2c, t_w2c, R_b2g, t_b2g = [], [], [], []
    for T_bf, T_cb in zip(flange_poses_T, board_poses_T):
        Rc, tc = se3.T_to_Rt(T_cb)            # world(board) -> cam == T_cam_board
        Rb, tb = se3.T_to_Rt(T_bf)            # T_base_flange, fed as base2gripper (NOT inverted)
        R_w2c.append(Rc)
        t_w2c.append(tc.reshape(3, 1))
        R_b2g.append(Rb)
        t_b2g.append(tb.reshape(3, 1))
    flag = getattr(cv2, f"CALIB_ROBOT_WORLD_HAND_EYE_{method}")
    R_b2w, t_b2w, R_g2c, t_g2c = cv2.calibrateRobotWorldHandEye(
        R_w2c, t_w2c, R_b2g, t_b2g, method=flag)
    T_base_camera = se3.invert_T(se3.Rt_to_T(R_g2c, t_g2c))    # invert(T_camera_base)
    T_flange_board = se3.invert_T(se3.Rt_to_T(R_b2w, t_b2w))   # invert(T_board_flange)
    return T_base_camera, T_flange_board


def cross_solver_spread(T_by_method: dict):
    """Max pairwise translation (mm) and rotation (deg) spread across solver results."""
    Ts = list(T_by_method.values())
    max_mm = 0.0
    max_deg = 0.0
    for i in range(len(Ts)):
        for j in range(i + 1, len(Ts)):
            dt = np.linalg.norm(Ts[i][:3, 3] - Ts[j][:3, 3]) * 1000.0
            dR = Ts[i][:3, :3] @ Ts[j][:3, :3].T
            max_mm = max(max_mm, dt)
            max_deg = max(max_deg, se3.rotation_angle_deg(dR))
    return float(max_mm), float(max_deg)


def axxb_residuals(T_base_camera, flange_poses_T, board_poses_T, T_flange_board=None):
    """Per-pose AX=XB consistency residual.

    With the recovered T_base_camera and the (constant) T_flange_board, the board's pose seen by the
    camera should equal T_camera_base @ T_base_flange @ T_flange_board. Residual = pose error between
    that predicted T_cam_board and the measured one. If T_flange_board is None it is estimated as the
    average of invert(T_base_camera@T_base_flange_i@?) ... so we estimate it from the data:
        T_flange_board_i = invert(T_base_flange_i) @ T_base_camera @ T_cam_board_i
    and use the mean; the spread of T_flange_board_i across poses IS the residual signal.
    """
    T_cam_base = se3.invert_T(T_base_camera)
    fb = []
    for T_bf, T_cb in zip(flange_poses_T, board_poses_T):
        fb.append(se3.invert_T(T_bf) @ T_base_camera @ T_cb)
    if T_flange_board is None:
        # robust mean: average translation, and a rotation close to all (use the first as ref)
        t_mean = np.mean([T[:3, 3] for T in fb], axis=0)
        T_flange_board = fb[0].copy()
        T_flange_board[:3, 3] = t_mean
    trans_mm, rot_deg = [], []
    for T_bf, T_cb in zip(flange_poses_T, board_poses_T):
        T_pred = T_cam_base @ T_bf @ T_flange_board     # predicted T_cam_board
        dt = np.linalg.norm(T_pred[:3, 3] - T_cb[:3, 3]) * 1000.0
        dR = T_pred[:3, :3] @ T_cb[:3, :3].T
        trans_mm.append(dt)
        rot_deg.append(se3.rotation_angle_deg(dR))
    return {
        "translation_mm_mean": float(np.mean(trans_mm)),
        "translation_mm_max": float(np.max(trans_mm)),
        "rotation_deg_mean": float(np.mean(rot_deg)),
        "rotation_deg_max": float(np.max(rot_deg)),
        "T_flange_board": T_flange_board,
    }
