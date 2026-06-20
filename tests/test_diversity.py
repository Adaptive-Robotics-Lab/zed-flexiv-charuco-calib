"""The anti-degeneracy gate must PASS a diverse set and FAIL the old coplanar one -- pure numpy."""
import numpy as np
from conftest import make_coplanar_flange_poses, make_diverse_flange_poses

from zfcc.config import DiversityGates
from zfcc.diversity import (
    assess_diversity,
    coplanarity_index,
    n_distinct_rotation_axes,
    rotation_axis_spread_deg,
)


def test_diverse_set_passes():
    poses = make_diverse_flange_poses(24)
    rep = assess_diversity(poses, DiversityGates())
    assert rep.verdict == "PASS", rep.reasons
    assert rep.rotation_axes >= 3
    assert rep.coplanarity_index > 0.04


def test_coplanar_set_fails_like_old_calibration():
    poses = make_coplanar_flange_poses(8)
    rep = assess_diversity(poses, DiversityGates())
    assert rep.verdict == "FAIL"
    # must cite at least one of the real degeneracy reasons
    text = " ".join(rep.reasons).lower()
    assert ("coplanar" in text or "rotation ax" in text or "poses" in text)


def test_coplanarity_index_zero_for_planar_points():
    pts = np.array([[0, 0, 0.3], [0.1, 0, 0.3], [0, 0.1, 0.3], [0.1, 0.1, 0.3]], dtype=float)
    assert coplanarity_index(pts) < 1e-6


def test_coplanarity_index_positive_for_volume():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 0.3, size=(20, 3))
    assert coplanarity_index(pts) > 0.1


def test_n_distinct_axes_counts_separated_axes():
    from conftest import R_axis_angle

    from zfcc import se3
    poses = [se3.Rt_to_T(R_axis_angle(a, 40), np.zeros(3))
             for a in ([1, 0, 0], [0, 1, 0], [0, 0, 1])]
    assert n_distinct_rotation_axes(poses) == 3


def test_rotation_axis_spread_low_for_single_axis():
    from conftest import R_axis_angle

    from zfcc import se3
    poses = [se3.Rt_to_T(R_axis_angle([0, 0, 1], d), np.zeros(3)) for d in (10, 20, 30, 40)]
    assert rotation_axis_spread_deg(poses) < 5.0
