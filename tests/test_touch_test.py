"""Touch-test geometry -- pure numpy, no hardware."""
import numpy as np
from conftest import R_axis_angle

from zfcc import se3
from zfcc.touch_test import board_corner_in_camera, camera_point_to_base, touch_error


def test_camera_point_to_base_matches_manual_transform():
    T_bc = se3.Rt_to_T(R_axis_angle([0, 1, 0], 90), np.array([0.5, 0.0, 0.3]))
    p_cam = np.array([0.1, -0.05, 0.6])
    p_base = camera_point_to_base(T_bc, p_cam)
    expect = T_bc[:3, :3] @ p_cam + T_bc[:3, 3]
    assert np.allclose(p_base, expect, atol=1e-12)


def test_board_corner_in_camera():
    objp = np.array([[0, 0, 0], [0.04, 0, 0], [0, 0.04, 0]], dtype=float)
    T_cam_board = se3.Rt_to_T(np.eye(3), np.array([0.0, 0.0, 0.5]))
    c = board_corner_in_camera(T_cam_board, objp, 1)
    assert np.allclose(c, [0.04, 0, 0.5], atol=1e-12)


def test_touch_error_is_euclidean_mm():
    r = touch_error([0.10, 0.20, 0.30], [0.103, 0.20, 0.296])
    assert abs(r.error_mm - 5.0) < 1e-6   # sqrt(3^2+0+4^2) mm
