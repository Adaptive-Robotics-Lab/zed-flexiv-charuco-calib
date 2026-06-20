"""End-to-end: build a synthetic session, save/reload, solve, expect PASS + recovered extrinsic."""
import numpy as np
import pytest
from conftest import make_coplanar_flange_poses, make_diverse_flange_poses, synth_board_poses

from zfcc import se3
from zfcc.config import DiversityGates, PassBars
from zfcc.session import CalibrationSession, Capture

cv2 = pytest.importorskip("cv2")
from zfcc.session import solve_session


def _board_corners():
    xs = np.linspace(0, 0.52, 14)
    ys = np.linspace(0, 0.32, 9)
    return np.array([[x, y, 0.0] for x in xs for y in ys], dtype=float)


def _session_from(flange, boards, root):
    s = CalibrationSession(root=str(root), zed_serial="SNTEST", image_size=[1280, 720])
    for i, (T_bf, T_cb) in enumerate(zip(flange, boards)):
        s.add(Capture(index=i, image_path=None,
                      flange_pose7=se3.T_to_pose7(T_bf).tolist(),
                      n_corners=80, pnp_rms_px=0.2, T_cam_board=T_cb.tolist(),
                      laplacian_var=150.0))
    return s


def test_save_load_roundtrip(tmp_path, gt_extrinsics):
    T_bc, T_fb = gt_extrinsics
    flange = make_diverse_flange_poses(18)
    boards = synth_board_poses(flange, T_bc, T_fb)
    s = _session_from(flange, boards, tmp_path / "sess")
    s.save()
    s2 = CalibrationSession.load(tmp_path / "sess")
    assert len(s2.captures) == 18
    assert s2.zed_serial == "SNTEST"
    assert np.allclose(s2.captures[0].T_cam_board_mat, boards[0], atol=1e-12)


def test_solve_session_passes_and_recovers(tmp_path, gt_extrinsics):
    T_bc, T_fb = gt_extrinsics
    flange = make_diverse_flange_poses(20)
    boards = synth_board_poses(flange, T_bc, T_fb, noise_m=0.0003, noise_deg=0.03, seed=5)
    s = _session_from(flange, boards, tmp_path / "sess")
    report = solve_session(s, _board_corners(), gates=DiversityGates(), bars=PassBars())
    assert report["verdict"] in ("PASS", "WARN")
    T = np.asarray(report["T_base_camera"], dtype=float)
    assert np.linalg.norm(T[:3, 3] - T_bc[:3, 3]) * 1000 < 5.0


def test_solve_session_refuses_degenerate(tmp_path, gt_extrinsics):
    T_bc, T_fb = gt_extrinsics
    flange = make_coplanar_flange_poses(8)
    boards = synth_board_poses(flange, T_bc, T_fb)
    s = _session_from(flange, boards, tmp_path / "sess")
    report = solve_session(s, _board_corners(), gates=DiversityGates(), bars=PassBars())
    assert report["verdict"] == "FAIL"
    assert "T_base_camera" not in report   # refused before solving
