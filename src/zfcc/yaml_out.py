"""Serialize a solved calibration to the drop-in ``T_base_zed2i.yaml`` ActAhead consumes, plus a
full provenance/validation report. The schema matches ActAhead's loader exactly:
``cfg["T_base_zed2i"]["matrix"]`` is the 4x4 base<-camera transform.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from . import se3

__all__ = ["build_calibration_doc", "write_calibration_yaml", "build_intrinsics_doc",
           "write_intrinsics_yaml"]

FRAME_CONVENTION = "T_A_B maps a point expressed in frame B into frame A: p_A = T_A_B @ p_B"


def build_calibration_doc(T_base_camera, *, serial=None, validation=None, source=None,
                          parent_frame="flexiv_world", child_frame="zed2i_camera_frame") -> dict:
    """Assemble the YAML document (a plain dict) for the solved extrinsic."""
    T = np.asarray(T_base_camera, dtype=float)
    T_inv = se3.invert_T(T)
    pose7 = se3.T_to_pose7(T)   # [x,y,z, qw,qx,qy,qz] -- handy for Flexiv-style consumers
    doc = {
        "T_base_zed2i": {
            "frame_convention": FRAME_CONVENTION,
            "parent_frame": parent_frame,
            "child_frame": child_frame,
            "matrix": [[float(v) for v in row] for row in T],
            "translation_xyz_m": [float(v) for v in T[:3, 3]],
            "quaternion_wxyz": [float(v) for v in pose7[3:]],
            "inverse_T_camera_base": [[float(v) for v in row] for row in T_inv],
        },
        "serial": serial,
        "source": source or "zed-flexiv-charuco-calib (eye-to-hand ChArUco)",
        "usage": ("p_base = T_base_zed2i.matrix @ [p_camera; 1]; load matrix at "
                  "['T_base_zed2i']['matrix'] and apply to camera-frame 3D points."),
    }
    if validation is not None:
        doc["validation"] = validation
    return doc


def write_calibration_yaml(path, T_base_camera, **kw) -> str:
    doc = build_calibration_doc(T_base_camera, **kw)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False), encoding="utf-8")
    return str(p)


def build_intrinsics_doc(result, *, serial=None, factory_audit=None) -> dict:
    K = np.asarray(result.K, dtype=float)
    doc = {
        "camera_matrix": {
            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "matrix": [[float(v) for v in row] for row in K],
        },
        "distortion": [float(v) for v in np.asarray(result.D).reshape(-1)],
        "image_size": list(result.image_size),
        "reprojection_rms_px": round(float(result.rms_px), 4),
        "mode": result.mode,
        "n_views": result.n_views,
        "serial": serial,
        "note": ("ZED left stream is rectified (distortion ~ 0); 'audit' mode confirms the factory K "
                 "rather than replacing it -- depth is computed against factory K, do not overwrite."),
    }
    if factory_audit is not None:
        doc["factory_audit"] = factory_audit
    return doc


def write_intrinsics_yaml(path, result, **kw) -> str:
    doc = build_intrinsics_doc(result, **kw)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False), encoding="utf-8")
    return str(p)
