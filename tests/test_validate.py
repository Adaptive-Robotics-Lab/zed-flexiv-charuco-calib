"""Validation utilities on synthetic geometry (handeye import pulls cv2 -> importorskip)."""
import numpy as np
import pytest
from conftest import make_diverse_flange_poses, synth_board_poses

from zfcc import se3

cv2 = pytest.importorskip("cv2")
from zfcc.validate import (
    base_frame_corner_error_m,
    leave_one_out,
    reject_outliers,
    reproject_board_to_base,
)


def _board_corners():
    xs = np.linspace(0, 0.52, 14)
    ys = np.linspace(0, 0.32, 9)
    return np.array([[x, y, 0.0] for x in xs for y in ys], dtype=float)


def test_leave_one_out_stable_for_clean_data(gt_extrinsics):
    T_bc, T_fb = gt_extrinsics
    flange = make_diverse_flange_poses(20)
    boards = synth_board_poses(flange, T_bc, T_fb)
    loo = leave_one_out(flange, boards)
    assert loo["origin_std_mm"] < 0.5
    assert loo["rotation_dev_deg_max"] < 0.05


def test_base_frame_corner_error_near_zero_for_truth(gt_extrinsics):
    T_bc, T_fb = gt_extrinsics
    flange = make_diverse_flange_poses(20)
    boards = synth_board_poses(flange, T_bc, T_fb)
    err = base_frame_corner_error_m(T_bc, flange, boards, _board_corners())
    assert err["corner_err_mm_max"] < 0.01


def test_reject_outliers_flags_bad_pnp(gt_extrinsics):
    T_bc, T_fb = gt_extrinsics
    flange = make_diverse_flange_poses(20)
    boards = synth_board_poses(flange, T_bc, T_fb)
    rms = [0.2] * len(flange)
    rms[5] = 3.5   # one blurry view
    keep, dropped = reject_outliers(flange, boards, rms, T_base_camera=T_bc,
                                    rms_px_max=1.0, axxb_mm_max=4.0)
    assert 5 not in keep
    assert any(i == 5 for i, _ in dropped)


def test_reproject_clouds_shape(gt_extrinsics):
    T_bc, T_fb = gt_extrinsics
    flange = make_diverse_flange_poses(6)
    boards = synth_board_poses(flange, T_bc, T_fb)
    clouds = reproject_board_to_base(T_bc, boards, _board_corners())
    assert len(clouds) == 6
    assert clouds[0].shape == (_board_corners().shape[0], 3)
