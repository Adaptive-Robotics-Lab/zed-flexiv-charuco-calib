"""Flexiv Rizon flange-pose reader (guarded import of ``flexivrdk``).

The ChArUco board is bolted to the FLANGE, so hand-eye needs ``T_base_flange`` -- NOT ``T_base_tcp``
(the TCP carries the gripper offset and would inject that offset straight into the extrinsic). Flexiv
poses are ``[x, y, z, qw, qx, qy, qz]`` (scalar-first quaternion), metres + unit quaternion. RDK
exposes the flange pose directly as ``RobotStates.flange_pose``; if a build lacks it, set the TCP to
the flange (zero tool) so ``tcp_pose == flange_pose`` and read that instead.

Two transports are supported, both optional:
  * "rdk":    direct ``flexivrdk.Robot`` on the robot PC.
  * "remote": the flexiv-control serve daemon over its socket (no flexivrdk needed locally).
"""
from __future__ import annotations

import numpy as np

from .config import RobotConfig

__all__ = ["FlangePoseReader", "pose7_is_valid"]


def pose7_is_valid(p) -> bool:
    p = np.asarray(p, dtype=float).reshape(-1)
    if p.shape[0] != 7:
        return False
    q = p[3:]
    return bool(abs(np.linalg.norm(q) - 1.0) < 1e-3)


class FlangePoseReader:
    """Read ``T_base_flange`` as a Flexiv pose7. Transport-agnostic façade."""

    def __init__(self, cfg: RobotConfig | None = None, transport: str = "rdk", client=None):
        self.cfg = cfg or RobotConfig()
        self.transport = transport
        self._client = client     # for "remote": a flexiv-control RemoteRobot; for "rdk": a Robot
        self._mode = None

    # ---- rdk transport -------------------------------------------------
    def _open_rdk(self):
        try:
            import flexivrdk
        except Exception as e:  # pragma: no cover - hardware path
            raise RuntimeError("flexivrdk not importable on this host") from e
        robot = flexivrdk.Robot(self.cfg.serial or self.cfg.host)
        if robot.fault():
            robot.ClearFault()
        robot.Enable()
        self._client = robot
        self._mode = "rdk"
        return self

    def _flange_rdk(self):
        st = self._client.states()
        pose = getattr(st, "flange_pose", None)
        if pose is None:
            pose = st.tcp_pose   # requires tool==flange; documented in the procedure
        return np.asarray(pose, dtype=float).reshape(7)

    # ---- remote transport ----------------------------------------------
    def _flange_remote(self):
        # flexiv-control RemoteRobot: prefer an explicit flange getter; fall back to state dict.
        c = self._client
        for name in ("get_flange_pose", "flange_pose"):
            fn = getattr(c, name, None)
            if callable(fn):
                return np.asarray(fn(), dtype=float).reshape(7)
        st = c.get_state()
        pose = st.get("flange_pose") or st.get("tcp_pose")
        return np.asarray(pose, dtype=float).reshape(7)

    # ---- public --------------------------------------------------------
    def open(self):
        if self.transport == "rdk" and self._client is None:
            self._open_rdk()
        else:
            self._mode = self.transport
        return self

    def read_flange_pose7(self) -> np.ndarray:
        if self._mode is None:
            self.open()
        p = self._flange_rdk() if self._mode == "rdk" else self._flange_remote()
        if not pose7_is_valid(p):
            raise RuntimeError(f"invalid flange pose7 read: {p}")
        return p

    def close(self):
        # remote client lifecycle is owned by the caller; rdk robot needs no explicit close here.
        self._client = None
