"""ZED 2i grabber (guarded import of ``pyzed``).

Importing this module never requires the ZED SDK; only constructing ``ZedCamera`` does. The left
stream is RECTIFIED and the factory K is read straight from the SDK -- that K is the one the depth
engine uses, so it is what every downstream transform must assume.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ZedConfig

__all__ = ["ZedFrame", "ZedCamera", "factory_K_from_sdk"]


@dataclass
class ZedFrame:
    bgr: np.ndarray
    gray: np.ndarray
    K: np.ndarray
    image_size: tuple[int, int]
    serial: str | None = None


def _sl():
    try:
        import pyzed.sl as sl
    except Exception as e:  # pragma: no cover - hardware path
        raise RuntimeError(
            "pyzed (ZED SDK) is not importable; install the ZED SDK + python API to capture frames"
        ) from e
    return sl


def factory_K_from_sdk(cam_info, sl) -> np.ndarray:
    """Pull the left-rectified factory intrinsics out of an opened camera's calibration."""
    calib = cam_info.camera_configuration.calibration_parameters
    fx, fy = calib.left_cam.fx, calib.left_cam.fy
    cx, cy = calib.left_cam.cx, calib.left_cam.cy
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=float)


class ZedCamera:
    """Minimal context-managed grabber. Disables self-calibration so K is frozen across opens."""

    def __init__(self, cfg: ZedConfig | None = None):
        self.cfg = cfg or ZedConfig()
        self._cam = None
        self._sl = None
        self.K = None
        self.image_size = None
        self.serial = self.cfg.serial

    def __enter__(self):
        sl = _sl()
        self._sl = sl
        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, self.cfg.resolution)
        init.depth_mode = getattr(sl.DEPTH_MODE, self.cfg.depth_mode)
        init.coordinate_units = sl.UNIT.METER
        init.depth_minimum_distance = self.cfg.depth_minimum_distance_m
        init.camera_disable_self_calib = bool(self.cfg.disable_self_calib)
        cam = sl.Camera()
        status = cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {status}")
        self._cam = cam
        info = cam.get_camera_information()
        self.K = factory_K_from_sdk(info, sl)
        res = info.camera_configuration.resolution
        self.image_size = (int(res.width), int(res.height))
        self.serial = str(info.serial_number)
        return self

    def __exit__(self, *exc):
        if self._cam is not None:
            self._cam.close()
        self._cam = None

    def grab(self) -> ZedFrame:
        sl = self._sl
        rt = sl.RuntimeParameters()
        if self._cam.grab(rt) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("ZED grab failed")
        mat = sl.Mat()
        self._cam.retrieve_image(mat, sl.VIEW.LEFT)   # rectified left
        import cv2

        bgra = mat.get_data()
        bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return ZedFrame(bgr=bgr, gray=gray, K=self.K.copy(),
                        image_size=self.image_size, serial=self.serial)

    def retrieve_point_cloud(self):
        """XYZ point cloud (metres, camera frame) aligned to the left image -- for touch targets."""
        sl = self._sl
        rt = sl.RuntimeParameters()
        if self._cam.grab(rt) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("ZED grab failed")
        xyz = sl.Mat()
        self._cam.retrieve_measure(xyz, sl.MEASURE.XYZ)
        return xyz.get_data()[:, :, :3]
