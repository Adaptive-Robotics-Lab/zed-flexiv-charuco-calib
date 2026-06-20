"""YAML drop-in schema -- pure (yaml + numpy). Guards the contract ActAhead's loader depends on."""
import numpy as np
import yaml
from conftest import R_axis_angle

from zfcc import se3
from zfcc.yaml_out import build_calibration_doc, write_calibration_yaml


def _T():
    return se3.Rt_to_T(R_axis_angle([0.2, 1, 0.1], 120.0), np.array([0.9, -0.3, 0.7]))


def test_doc_schema_has_actahead_matrix_key():
    doc = build_calibration_doc(_T(), serial="SN123")
    assert "T_base_zed2i" in doc
    assert "matrix" in doc["T_base_zed2i"]
    M = np.asarray(doc["T_base_zed2i"]["matrix"], dtype=float)
    assert M.shape == (4, 4)
    assert np.allclose(M, _T(), atol=1e-12)
    assert doc["serial"] == "SN123"


def test_inverse_is_consistent():
    doc = build_calibration_doc(_T())
    M = np.asarray(doc["T_base_zed2i"]["matrix"], dtype=float)
    Minv = np.asarray(doc["T_base_zed2i"]["inverse_T_camera_base"], dtype=float)
    assert np.allclose(M @ Minv, np.eye(4), atol=1e-9)


def test_quaternion_and_translation_match_matrix():
    doc = build_calibration_doc(_T())["T_base_zed2i"]
    q = np.asarray(doc["quaternion_wxyz"], dtype=float)
    t = np.asarray(doc["translation_xyz_m"], dtype=float)
    T2 = se3.pose7_to_T(np.concatenate([t, q]))
    assert np.allclose(T2, _T(), atol=1e-9)


def test_write_and_reload(tmp_path):
    p = write_calibration_yaml(tmp_path / "T_base_zed2i.yaml", _T(), serial="SNX",
                               validation={"verdict": "PASS"})
    doc = yaml.safe_load(open(p, encoding="utf-8"))
    M = np.asarray(doc["T_base_zed2i"]["matrix"], dtype=float)
    assert np.allclose(M, _T(), atol=1e-12)
    assert doc["validation"]["verdict"] == "PASS"
