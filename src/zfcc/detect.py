"""Per-image ChArUco detection, board-pose PnP, and per-image quality metrics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import se3

__all__ = ["Detection", "detect_charuco", "board_pose_pnp", "laplacian_var", "corner_coverage_bins"]


@dataclass
class Detection:
    charuco_corners: np.ndarray  # (N,1,2) float32 image points
    charuco_ids: np.ndarray      # (N,1) int
    n_corners: int
    laplacian_var: float         # focus measure (higher = sharper)

    @property
    def ok(self) -> bool:
        return self.n_corners >= 4


def laplacian_var(gray) -> float:
    import cv2

    return float(cv2.Laplacian(np.asarray(gray), cv2.CV_64F).var())


def detect_charuco(gray, board, detector) -> Detection:
    cc, ci, _mc, _mi = detector.detectBoard(np.asarray(gray))
    n = 0 if ci is None else int(len(ci))
    return Detection(charuco_corners=cc, charuco_ids=ci, n_corners=n,
                     laplacian_var=laplacian_var(gray))


def board_pose_pnp(det: Detection, board, K, D=None, min_corners: int = 6):
    """Estimate T_cam_board for one detection via IPPE + LM refine.

    Returns (T_cam_board 4x4, reproj_rms_px). Raises if too few corners.
    K is 3x3; D is the distortion vector (zeros for the rectified ZED left stream).
    """
    import cv2

    if det.charuco_ids is None or det.n_corners < min_corners:
        raise ValueError(f"too few charuco corners for PnP: {det.n_corners} < {min_corners}")
    K = np.asarray(K, dtype=float)
    D = np.zeros((5, 1), dtype=float) if D is None else np.asarray(D, dtype=float).reshape(-1, 1)
    objp, imgp = board.matchImagePoints(det.charuco_corners, det.charuco_ids)
    objp = np.asarray(objp, dtype=np.float32).reshape(-1, 1, 3)
    imgp = np.asarray(imgp, dtype=np.float32).reshape(-1, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, D, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed for board pose")
    rvec, tvec = cv2.solvePnPRefineLM(objp, imgp, K, D, rvec, tvec)
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - imgp.reshape(-1, 2)) ** 2, axis=1))))
    return se3.rvec_tvec_to_T(rvec, tvec), rms


def corner_coverage_bins(det: Detection, image_size, grid=(3, 3)) -> np.ndarray:
    """Count detected corners per image cell (default 3x3) -> coverage heatmap for a gate."""
    w, h = int(image_size[0]), int(image_size[1])
    gx, gy = grid
    bins = np.zeros((gy, gx), dtype=int)
    if det.charuco_corners is None:
        return bins
    pts = np.asarray(det.charuco_corners, dtype=float).reshape(-1, 2)
    for x, y in pts:
        cx = min(gx - 1, max(0, int(x / max(w, 1) * gx)))
        cy = min(gy - 1, max(0, int(y / max(h, 1) * gy)))
        bins[cy, cx] += 1
    return bins
