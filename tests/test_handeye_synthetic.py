"""The correctness keystone: synthesize a known eye-to-hand geometry, run the solver, and assert it
recovers the ground-truth T_base_camera. This is what proves the eye-to-hand INVERSION sign is right
-- a flipped inversion would still return a clean matrix, just the wrong one, so we test the value.
"""
import numpy as np
import pytest
from conftest import make_diverse_flange_poses, synth_board_poses

from zfcc import se3
from zfcc.handeye import axxb_residuals, cross_solver_spread, solve_eye_to_hand, solve_robot_world

cv2 = pytest.importorskip("cv2")


def test_recovers_ground_truth_noisefree(gt_extrinsics):
    T_bc_gt, T_fb_gt = gt_extrinsics
    flange = make_diverse_flange_poses(24)
    boards = synth_board_poses(flange, T_bc_gt, T_fb_gt)
    res = solve_eye_to_hand(flange, boards)
    T = res.T_primary
    assert np.linalg.norm(T[:3, 3] - T_bc_gt[:3, 3]) * 1000 < 0.5   # < 0.5 mm
    assert se3.rotation_angle_deg(T[:3, :3] @ T_bc_gt[:3, :3].T) < 0.05  # < 0.05 deg


def test_all_five_solvers_agree(gt_extrinsics):
    T_bc_gt, T_fb_gt = gt_extrinsics
    flange = make_diverse_flange_poses(24)
    boards = synth_board_poses(flange, T_bc_gt, T_fb_gt)
    res = solve_eye_to_hand(flange, boards)
    assert set(res.T_base_camera) == {"TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"}
    mm, deg = cross_solver_spread(res.T_base_camera)
    assert mm < 0.5 and deg < 0.05


def test_inversion_sign_guard(gt_extrinsics):
    """If someone removed the invert_T in solve_eye_to_hand, feeding NON-inverted flange poses would
    'recover' something far from ground truth. Confirm the wrong convention is detectably wrong."""
    T_bc_gt, T_fb_gt = gt_extrinsics
    flange = make_diverse_flange_poses(24)
    boards = synth_board_poses(flange, T_bc_gt, T_fb_gt)
    # deliberately pass already-inverted flange poses -> double inversion -> wrong answer
    wrong = [se3.invert_T(T) for T in flange]
    res = solve_eye_to_hand(wrong, boards)
    err_mm = np.linalg.norm(res.T_primary[:3, 3] - T_bc_gt[:3, 3]) * 1000
    assert err_mm > 5.0   # the correct convention (test above) gets < 0.5 mm


def test_robot_world_crosscheck_recovers_both(gt_extrinsics):
    T_bc_gt, T_fb_gt = gt_extrinsics
    flange = make_diverse_flange_poses(24)
    boards = synth_board_poses(flange, T_bc_gt, T_fb_gt)
    T_bc, T_fb = solve_robot_world(flange, boards)
    # noise-free synthetic data: the robot-world solver must recover both transforms exactly
    assert np.linalg.norm(T_bc[:3, 3] - T_bc_gt[:3, 3]) * 1000 < 0.01
    assert se3.rotation_angle_deg(T_bc[:3, :3] @ T_bc_gt[:3, :3].T) < 0.01
    assert np.linalg.norm(T_fb[:3, 3] - T_fb_gt[:3, 3]) * 1000 < 0.01


def test_axxb_residual_small_for_truth(gt_extrinsics):
    T_bc_gt, T_fb_gt = gt_extrinsics
    flange = make_diverse_flange_poses(24)
    boards = synth_board_poses(flange, T_bc_gt, T_fb_gt)
    res = axxb_residuals(T_bc_gt, flange, boards, T_flange_board=T_fb_gt)
    assert res["translation_mm_max"] < 1e-6
    assert res["rotation_deg_max"] < 1e-3   # geodesic angle floors at ~sqrt(eps) for an exact match


def test_noise_degrades_gracefully(gt_extrinsics):
    T_bc_gt, T_fb_gt = gt_extrinsics
    flange = make_diverse_flange_poses(24)
    boards = synth_board_poses(flange, T_bc_gt, T_fb_gt, noise_m=0.0005, noise_deg=0.05, seed=11)
    res = solve_eye_to_hand(flange, boards)
    # realistic sub-mm/sub-deg detection noise -> still well under a few mm
    assert np.linalg.norm(res.T_primary[:3, 3] - T_bc_gt[:3, 3]) * 1000 < 5.0
