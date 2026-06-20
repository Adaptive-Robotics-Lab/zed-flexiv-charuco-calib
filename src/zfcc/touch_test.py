"""Physical end-to-end accuracy check ("touch test").

A reprojection number can look great and the grasp still miss, because reprojection only checks the
camera<->board leg. The touch test checks the WHOLE chain the robot actually uses: pick a target the
camera sees (a ChArUco corner, or any 3D point you click), transform it base<-camera with the solved
T_base_camera, command the TCP there, and measure the real miss with a ruler / dial indicator. This
is the number that predicts whether grasps land -- and the metric that would have caught the old
~1.5-2cm height error.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import se3

__all__ = ["TouchTarget", "camera_point_to_base", "board_corner_in_camera", "TouchResult"]


@dataclass
class TouchTarget:
    name: str
    p_camera_m: np.ndarray       # 3D point in the ZED camera frame
    p_base_pred_m: np.ndarray    # predicted base-frame point (filled by camera_point_to_base)


@dataclass
class TouchResult:
    target: str
    commanded_base_m: np.ndarray
    measured_base_m: np.ndarray
    error_mm: float


def camera_point_to_base(T_base_camera, p_camera_m) -> np.ndarray:
    """Transform a camera-frame 3D point into the robot base frame using the solved extrinsic."""
    p = np.asarray(p_camera_m, dtype=float).reshape(3)
    R, t = se3.T_to_Rt(np.asarray(T_base_camera, dtype=float))
    return R @ p + t


def board_corner_in_camera(T_cam_board, board_object_points, corner_index: int) -> np.ndarray:
    """Camera-frame coordinates of a specific board corner, for use as a touch target."""
    objp = np.asarray(board_object_points, dtype=float).reshape(-1, 3)
    c = objp[corner_index]
    R, t = se3.T_to_Rt(np.asarray(T_cam_board, dtype=float))
    return R @ c + t


def touch_error(commanded_base_m, measured_base_m) -> TouchResult:
    """Record one touch: where we told the TCP to go vs where the ruler says it landed."""
    c = np.asarray(commanded_base_m, dtype=float).reshape(3)
    m = np.asarray(measured_base_m, dtype=float).reshape(3)
    return TouchResult(target="touch", commanded_base_m=c, measured_base_m=m,
                       error_mm=float(np.linalg.norm(c - m) * 1000.0))
