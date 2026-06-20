"""Shared synthetic fixtures for the math-core tests (no hardware, no cv2 needed to import)."""
import numpy as np
import pytest

from zfcc import se3


def rand_R(rng):
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def R_axis_angle(axis, deg):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    th = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def make_diverse_flange_poses(n=24, seed=7):
    """A well-conditioned eye-to-hand pose set: many rotation axes, varied positions/depths."""
    rng = np.random.default_rng(seed)
    axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, -1, 0]]
    poses = []
    for i in range(n):
        axis = axes[i % len(axes)]
        deg = 25 + 50 * rng.random()
        R = R_axis_angle(axis, deg) @ rand_R(rng) if i % 3 == 0 else R_axis_angle(axis, deg)
        t = np.array([0.4, 0.0, 0.4]) + rng.uniform(-0.15, 0.15, size=3)
        poses.append(se3.Rt_to_T(R, t))
    return poses


def make_coplanar_flange_poses(n=8, seed=3):
    """A degenerate set mimicking the OLD calibration: tiny rotations, positions on a plane (z fixed)."""
    rng = np.random.default_rng(seed)
    poses = []
    for _ in range(n):
        R = R_axis_angle([0, 0, 1], rng.uniform(-8, 8))  # all about ~same axis, small angle
        t = np.array([0.4 + rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), 0.35])  # fixed z plane
        poses.append(se3.Rt_to_T(R, t))
    return poses


@pytest.fixture
def gt_extrinsics():
    """Ground-truth (T_base_camera, T_flange_board) for synthetic round-trips."""
    T_base_camera = se3.Rt_to_T(R_axis_angle([0.2, 1, 0.1], 120.0), np.array([0.9, -0.3, 0.7]))
    T_flange_board = se3.Rt_to_T(R_axis_angle([1, 0.1, 0.2], 15.0), np.array([0.0, 0.0, 0.12]))
    return T_base_camera, T_flange_board


def synth_board_poses(flange_poses, T_base_camera, T_flange_board, noise_m=0.0, noise_deg=0.0, seed=0):
    """T_cam_board_i = inv(T_base_camera) @ T_base_flange_i @ T_flange_board, with optional noise."""
    rng = np.random.default_rng(seed)
    T_cam_base = se3.invert_T(T_base_camera)
    out = []
    for T_bf in flange_poses:
        T_cb = T_cam_base @ T_bf @ T_flange_board
        if noise_m or noise_deg:
            dR = R_axis_angle(rng.normal(size=3) + 1e-9, rng.normal() * noise_deg)
            T_cb = T_cb.copy()
            T_cb[:3, :3] = dR @ T_cb[:3, :3]
            T_cb[:3, 3] += rng.normal(size=3) * noise_m
        out.append(T_cb)
    return out
