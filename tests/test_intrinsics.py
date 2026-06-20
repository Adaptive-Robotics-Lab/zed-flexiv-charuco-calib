"""Intrinsics: confirm ChArUco recovers a known K (free solve) and that audit mode reprojects a
correct K with ~0 error and does NOT change it. Uses synthetic projection -- no rendering/hardware.
"""
import numpy as np
import pytest

from zfcc.config import BoardConfig

cv2 = pytest.importorskip("cv2")
from zfcc.board import make_board
from zfcc.detect import Detection
from zfcc.intrinsics import audit_against_factory, calibrate_intrinsics

IMG = (1280, 720)
K_TRUE = np.array([[700.0, 0, 640.0], [0, 700.0, 360.0], [0, 0, 1.0]])


def _rodrigues(ax, deg):
    r = np.asarray(ax, float)
    r = r / np.linalg.norm(r) * np.radians(deg)
    R, _ = cv2.Rodrigues(r)
    return R


def _make_views(board, n=16, seed=0):
    rng = np.random.default_rng(seed)
    objp = np.asarray(board.getChessboardCorners(), dtype=np.float32).reshape(-1, 3)
    N = objp.shape[0]
    cx, cy = objp[:, 0].mean(), objp[:, 1].mean()
    dets = []
    tries = 0
    while len(dets) < n and tries < 400:
        tries += 1
        R = _rodrigues([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-0.3, 0.3)],
                       rng.uniform(5, 22))
        rvec, _ = cv2.Rodrigues(R)
        Z = rng.uniform(0.7, 1.0)
        tvec = np.array([-cx + rng.uniform(-0.05, 0.05), -cy + rng.uniform(-0.05, 0.05), Z])
        img, _ = cv2.projectPoints(objp.reshape(-1, 1, 3), rvec, tvec.reshape(3, 1), K_TRUE, None)
        pts = img.reshape(-1, 2)
        if pts[:, 0].min() < 5 or pts[:, 0].max() > IMG[0] - 5:
            continue
        if pts[:, 1].min() < 5 or pts[:, 1].max() > IMG[1] - 5:
            continue
        det = Detection(charuco_corners=img.astype(np.float32),
                        charuco_ids=np.arange(N, dtype=np.int32).reshape(-1, 1),
                        n_corners=N, laplacian_var=120.0)
        dets.append(det)
    assert len(dets) >= n, f"only built {len(dets)} in-bounds views"
    return dets


def test_free_solve_recovers_known_K():
    board = make_board(BoardConfig())
    dets = _make_views(board, n=16)
    res = calibrate_intrinsics(dets, board, IMG, K0=K_TRUE, mode="free")
    assert res.rms_px < 0.05
    assert abs(res.K[0, 0] - 700) < 1.0 and abs(res.K[1, 1] - 700) < 1.0
    assert abs(res.K[0, 2] - 640) < 1.0 and abs(res.K[1, 2] - 360) < 1.0


def test_audit_mode_reprojects_zero_and_keeps_K_fixed():
    board = make_board(BoardConfig())
    dets = _make_views(board, n=16)
    res = calibrate_intrinsics(dets, board, IMG, K0=K_TRUE, mode="audit")
    assert res.mode == "audit"
    assert res.rms_px < 0.05
    assert np.allclose(res.K, K_TRUE, atol=1e-6)   # fixed: not optimized away


def test_audit_against_factory_small_delta():
    board = make_board(BoardConfig())
    dets = _make_views(board, n=16)
    free = calibrate_intrinsics(dets, board, IMG, K0=K_TRUE, mode="free")
    d = audit_against_factory(free, K_TRUE)
    assert abs(d["d_fx_px"]) < 1.0 and abs(d["d_cx_px"]) < 1.0


def test_audit_requires_K0():
    board = make_board(BoardConfig())
    dets = _make_views(board, n=16)
    with pytest.raises(ValueError):
        calibrate_intrinsics(dets, board, IMG, K0=None, mode="audit")
