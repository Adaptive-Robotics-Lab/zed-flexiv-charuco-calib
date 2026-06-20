import numpy as np
import pytest

from zfcc import se3


def _rand_R(rng):
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def test_quat_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        R = _rand_R(rng)
        q = se3.R_to_quat_wxyz(R)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-9
        R2 = se3.quat_wxyz_to_R(q)
        assert np.allclose(R, R2, atol=1e-9)


def test_pose7_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(100):
        t = rng.normal(size=3)
        q = se3.R_to_quat_wxyz(_rand_R(rng))
        pose7 = np.concatenate([t, q])
        T = se3.pose7_to_T(pose7)
        back = se3.T_to_pose7(T)
        assert np.allclose(back[:3], t, atol=1e-9)
        # quaternion equal up to sign
        assert np.allclose(back[3:], q, atol=1e-7) or np.allclose(back[3:], -q, atol=1e-7)


def test_invert_is_true_inverse():
    rng = np.random.default_rng(2)
    for _ in range(100):
        T = se3.Rt_to_T(_rand_R(rng), rng.normal(size=3))
        Ti = se3.invert_T(T)
        assert np.allclose(T @ Ti, np.eye(4), atol=1e-9)
        assert np.allclose(Ti @ T, np.eye(4), atol=1e-9)
        # matches dense inverse but is orthonormal-exact
        assert np.allclose(Ti, np.linalg.inv(T), atol=1e-9)


def test_rotation_angle_and_axis():
    # 90 deg about z
    Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    assert abs(se3.rotation_angle_deg(Rz) - 90.0) < 1e-6
    ax = se3.rotation_axis(Rz)
    assert np.allclose(np.abs(ax), [0, 0, 1], atol=1e-6)


def test_rotation_axis_near_pi():
    # 180 deg about x: trace-based axis vanishes, must fall back to symmetric part
    Rx = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
    assert abs(se3.rotation_angle_deg(Rx) - 180.0) < 1e-6
    ax = se3.rotation_axis(Rx)
    assert np.allclose(np.abs(ax), [1, 0, 0], atol=1e-6)


def test_zero_quat_raises():
    with pytest.raises(ValueError):
        se3.quat_wxyz_to_R([0, 0, 0, 0])
