"""SE(3) bookkeeping — the ONE place rigid-transform conversions and inversions live.

Conventions (shared with flexiv_control and ActAhead):
  * Quaternions are scalar-first ``(w, x, y, z)`` (Flexiv / RDK convention).
  * A pose7 vector is ``[x, y, z, qw, qx, qy, qz]`` (the Flexiv ``tcp_pose`` / ``flange_pose`` layout).
  * ``T_A_B`` is a 4x4 homogeneous transform that maps a point expressed in frame B into frame A
    (i.e. ``p_A = T_A_B @ p_B``). This matches ActAhead's ``T_base_zed2i`` file.

This module is pure numpy so the math core and its tests need no OpenCV, pyzed or flexiv_control.
OpenCV rvec/tvec helpers import cv2 lazily, only where used.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "quat_wxyz_to_R",
    "R_to_quat_wxyz",
    "pose7_to_T",
    "T_to_pose7",
    "invert_T",
    "T_to_Rt",
    "Rt_to_T",
    "rvec_tvec_to_T",
    "T_to_rvec_tvec",
    "rotation_angle_deg",
    "rotation_axis",
]


def quat_wxyz_to_R(q) -> np.ndarray:
    """Scalar-first unit quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = (float(v) for v in q)
    n = (w * w + x * x + y * y + z * z) ** 0.5
    if n < 1e-12:
        raise ValueError("zero-norm quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def R_to_quat_wxyz(R) -> np.ndarray:
    """3x3 rotation matrix -> scalar-first unit quaternion (w, x, y, z).

    Shepperd's method (numerically stable across all rotations)."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    q /= np.linalg.norm(q)
    if q[0] < 0:  # canonical hemisphere (w >= 0)
        q = -q
    return q


def pose7_to_T(pose7) -> np.ndarray:
    """Flexiv pose ``[x, y, z, qw, qx, qy, qz]`` -> 4x4 homogeneous transform."""
    p = np.asarray(pose7, dtype=float).reshape(7)
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_R(p[3:7])
    T[:3, 3] = p[:3]
    return T


def T_to_pose7(T) -> np.ndarray:
    """4x4 homogeneous transform -> Flexiv pose ``[x, y, z, qw, qx, qy, qz]``."""
    T = np.asarray(T, dtype=float)
    q = R_to_quat_wxyz(T[:3, :3])
    return np.array([T[0, 3], T[1, 3], T[2, 3], q[0], q[1], q[2], q[3]], dtype=float)


def invert_T(T) -> np.ndarray:
    """Inverse of a rigid 4x4 transform (R.T, -R.T t) -- never use np.linalg.inv for SE(3)."""
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def T_to_Rt(T):
    """4x4 -> (R 3x3, t 3,)."""
    T = np.asarray(T, dtype=float)
    return T[:3, :3].copy(), T[:3, 3].copy()


def Rt_to_T(R, t) -> np.ndarray:
    """(R 3x3, t 3,) -> 4x4."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def rvec_tvec_to_T(rvec, tvec) -> np.ndarray:
    """OpenCV (rvec, tvec) -> 4x4."""
    import cv2

    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
    return Rt_to_T(R, np.asarray(tvec, dtype=float).reshape(3))


def T_to_rvec_tvec(T):
    """4x4 -> OpenCV (rvec 3x1, tvec 3x1)."""
    import cv2

    T = np.asarray(T, dtype=float)
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return rvec.reshape(3, 1), T[:3, 3].reshape(3, 1)


def rotation_angle_deg(R) -> float:
    """Geodesic rotation angle of a 3x3 rotation matrix, in degrees."""
    R = np.asarray(R, dtype=float)
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def rotation_axis(R) -> np.ndarray:
    """Unit rotation axis of a 3x3 rotation matrix (zero vector for ~identity)."""
    R = np.asarray(R, dtype=float)
    ang = np.radians(rotation_angle_deg(R))
    if ang < 1e-6:
        return np.zeros(3)
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=float)
    n = np.linalg.norm(ax)
    if n < 1e-9:  # angle near pi: axis from the symmetric part
        w, V = np.linalg.eigh((R + np.eye(3)) / 2.0)
        return V[:, int(np.argmax(w))]
    return ax / n
