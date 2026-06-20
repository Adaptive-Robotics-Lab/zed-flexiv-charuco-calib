"""Validation of a solved extrinsic: leave-one-out stability, per-pose outlier rejection, and a
3D board-corner reprojection error expressed in metres in the robot base frame.

These turn "the solver returned a matrix" into "the matrix is trustworthy" -- the difference between
the old coplanar calibration (which a solver also happily returned) and a calibration you can grasp on.
"""
from __future__ import annotations

import numpy as np

from . import se3
from .handeye import axxb_residuals, solve_eye_to_hand

__all__ = ["leave_one_out", "reject_outliers", "reproject_board_to_base", "base_frame_corner_error_m"]


def leave_one_out(flange_poses_T, board_poses_T, method="DANIILIDIS"):
    """Re-solve N times dropping one pose each; report the spread of T_base_camera origins.

    A stable calibration barely moves when any single pose is removed. Large spread => one pose is
    leveraging the fit (bad detection / wrong pairing) or the set is too small/degenerate.
    """
    n = len(flange_poses_T)
    origins, rots_ref = [], None
    for i in range(n):
        f = [T for j, T in enumerate(flange_poses_T) if j != i]
        b = [T for j, T in enumerate(board_poses_T) if j != i]
        res = solve_eye_to_hand(f, b, methods=(method,))
        T = res.T_primary
        origins.append(T[:3, 3])
        if rots_ref is None:
            rots_ref = T[:3, :3]
    origins = np.asarray(origins)
    rot_dev = []
    for i in range(n):
        f = [T for j, T in enumerate(flange_poses_T) if j != i]
        b = [T for j, T in enumerate(board_poses_T) if j != i]
        T = solve_eye_to_hand(f, b, methods=(method,)).T_primary
        rot_dev.append(se3.rotation_angle_deg(T[:3, :3] @ rots_ref.T))
    return {
        "origin_std_mm": float(np.linalg.norm(np.std(origins, axis=0)) * 1000.0),
        "origin_ptp_mm": float(np.max(np.linalg.norm(origins - origins.mean(0), axis=1)) * 1000.0),
        "rotation_dev_deg_max": float(np.max(rot_dev)),
        "n": n,
    }


def reject_outliers(flange_poses_T, board_poses_T, pnp_rms_px, T_base_camera=None,
                    rms_px_max=1.0, axxb_mm_max=4.0):
    """Flag poses whose per-frame PnP RMS is high or whose AX=XB residual is an outlier.

    Returns (keep_indices, dropped) where dropped lists (index, reason).
    """
    keep, dropped = [], []
    if T_base_camera is not None:
        res = axxb_residuals(T_base_camera, flange_poses_T, board_poses_T)
        T_fb = res["T_flange_board"]
        T_cam_base = se3.invert_T(T_base_camera)
        per_mm = []
        for T_bf, T_cb in zip(flange_poses_T, board_poses_T):
            T_pred = T_cam_base @ T_bf @ T_fb
            per_mm.append(np.linalg.norm(T_pred[:3, 3] - T_cb[:3, 3]) * 1000.0)
    else:
        per_mm = [0.0] * len(flange_poses_T)
    for i, (rms, mm) in enumerate(zip(pnp_rms_px, per_mm)):
        if rms is not None and rms > rms_px_max:
            dropped.append((i, f"pnp_rms {rms:.2f}px > {rms_px_max}"))
        elif mm > axxb_mm_max:
            dropped.append((i, f"axxb {mm:.1f}mm > {axxb_mm_max}"))
        else:
            keep.append(i)
    return keep, dropped


def reproject_board_to_base(T_base_camera, board_poses_T, board_object_points):
    """Map the board corners (object frame) through each measured camera pose into the base frame.

    If the calibration and pairing are consistent, the SAME physical corner should land at nearly the
    same base-frame point across all views (the board is rigidly on the flange, but its base-frame
    location changes per pose -- so we instead compare against the flange-predicted location). This
    helper returns the per-view base-frame corner clouds for inspection/plotting.
    """
    objp = np.asarray(board_object_points, dtype=float).reshape(-1, 3)
    clouds = []
    for T_cb in board_poses_T:
        T_base_board = T_base_camera @ T_cb
        pts = (T_base_board[:3, :3] @ objp.T + T_base_board[:3, 3:4]).T
        clouds.append(pts)
    return clouds


def base_frame_corner_error_m(T_base_camera, flange_poses_T, board_poses_T, board_object_points):
    """The headline metric: with the board rigidly on the flange, predict each board corner's
    base-frame position two ways and report the disagreement in mm.

      way A (camera): T_base_camera @ T_cam_board @ corner
      way B (robot) : T_base_flange  @ T_flange_board @ corner   (T_flange_board estimated)
    """
    res = axxb_residuals(T_base_camera, flange_poses_T, board_poses_T)
    T_fb = res["T_flange_board"]
    objp = np.asarray(board_object_points, dtype=float).reshape(-1, 3)
    errs = []
    for T_bf, T_cb in zip(flange_poses_T, board_poses_T):
        A = T_base_camera @ T_cb
        B = T_bf @ T_fb
        a = (A[:3, :3] @ objp.T + A[:3, 3:4]).T
        b = (B[:3, :3] @ objp.T + B[:3, 3:4]).T
        errs.append(np.linalg.norm(a - b, axis=1))
    errs = np.concatenate(errs)
    return {
        "corner_err_mm_mean": float(np.mean(errs) * 1000.0),
        "corner_err_mm_max": float(np.max(errs) * 1000.0),
        "corner_err_mm_p95": float(np.percentile(errs, 95) * 1000.0),
    }
