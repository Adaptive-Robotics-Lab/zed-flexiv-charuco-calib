"""Camera-intrinsics calibration from ChArUco views.

YES -- a ChArUco board recovers the full pinhole model (fx, fy, cx, cy + distortion), exactly like a
plain chessboard but robust to partial/occluded views because each detected corner carries its ID.
The modern path (``cv2.aruco.calibrateCameraCharuco`` was removed in 4.7) is:
    board.matchImagePoints(corners, ids) -> (objpoints, imgpoints) per view
    cv2.calibrateCamera(objpoints, imgpoints, image_size, K0, D0, flags=...)

IMPORTANT for the ZED 2i: its SDK left stream is already RECTIFIED (D == 0) against a factory-
calibrated K. We therefore run intrinsics in AUDIT mode by default: seed K0 with the factory matrix,
FIX principal point + focal length, and only confirm the reprojection RMS is small and that a free
solve lands close to factory. We do NOT overwrite the ZED's K -- the depth stream is computed against
the factory model, so a hand-rolled K would desync RGB and depth. The free solve is reported for
transparency.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["IntrinsicsResult", "calibrate_intrinsics", "audit_against_factory"]


@dataclass
class IntrinsicsResult:
    K: np.ndarray
    D: np.ndarray
    rms_px: float
    image_size: tuple[int, int]
    per_view_rms_px: list
    n_views: int
    mode: str               # "audit" (fixed) or "free"

    def as_dict(self) -> dict:
        return {
            "K": self.K.tolist(),
            "D": self.D.reshape(-1).tolist(),
            "rms_px": round(self.rms_px, 4),
            "image_size": list(self.image_size),
            "n_views": self.n_views,
            "mode": self.mode,
            "per_view_rms_px": [round(float(v), 4) for v in self.per_view_rms_px],
        }


def _per_view_rms(objpoints, imgpoints, rvecs, tvecs, K, D):
    import cv2

    out = []
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
        e = proj.reshape(-1, 2) - imgp.reshape(-1, 2)
        out.append(float(np.sqrt(np.mean(np.sum(e ** 2, axis=1)))))
    return out


def calibrate_intrinsics(detections, board, image_size, K0=None, mode="audit",
                         min_corners: int = 8) -> IntrinsicsResult:
    """Solve (or audit) intrinsics from a list of Detection objects.

    mode="audit": requires K0 (factory); fixes focal length + principal point + zero distortion,
                  so the only free parameters are the per-view extrinsics -> this yields the
                  reprojection RMS of the factory model on real data (the number that matters).
    mode="free":  full unconstrained solve, K0 only seeds the optimizer (or is computed if None).
    """
    import cv2

    objpoints, imgpoints = [], []
    for det in detections:
        if det.charuco_ids is None or det.n_corners < min_corners:
            continue
        objp, imgp = board.matchImagePoints(det.charuco_corners, det.charuco_ids)
        if objp is None or len(objp) < min_corners:
            continue
        objpoints.append(np.asarray(objp, dtype=np.float32).reshape(-1, 1, 3))
        imgpoints.append(np.asarray(imgp, dtype=np.float32).reshape(-1, 1, 2))
    if len(objpoints) < 3:
        raise ValueError(f"need >=3 usable views for intrinsics, got {len(objpoints)}")

    w, h = int(image_size[0]), int(image_size[1])
    if mode == "audit":
        if K0 is None:
            raise ValueError("audit mode requires the factory K0")
        K0 = np.asarray(K0, dtype=float)
        D0 = np.zeros((5, 1), dtype=float)
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT
                 | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_ZERO_TANGENT_DIST
                 | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, (w, h), K0.copy(), D0.copy(), flags=flags)
    else:
        flags = cv2.CALIB_RATIONAL_MODEL if False else 0
        guess = None if K0 is None else np.asarray(K0, dtype=float).copy()
        if guess is not None:
            flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, (w, h), guess, None, flags=flags)

    per_view = _per_view_rms(objpoints, imgpoints, rvecs, tvecs, K, D)
    return IntrinsicsResult(K=np.asarray(K), D=np.asarray(D), rms_px=float(rms),
                            image_size=(w, h), per_view_rms_px=per_view,
                            n_views=len(objpoints), mode=mode)


def audit_against_factory(free: IntrinsicsResult, factory_K) -> dict:
    """Compare a FREE intrinsic solve to the factory K -- a sanity bound on the rectified stream."""
    Kf = np.asarray(factory_K, dtype=float)
    Kc = np.asarray(free.K, dtype=float)
    return {
        "d_fx_px": float(Kc[0, 0] - Kf[0, 0]),
        "d_fy_px": float(Kc[1, 1] - Kf[1, 1]),
        "d_cx_px": float(Kc[0, 2] - Kf[0, 2]),
        "d_cy_px": float(Kc[1, 2] - Kf[1, 2]),
        "free_rms_px": round(free.rms_px, 4),
    }
