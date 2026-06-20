"""A calibration session: the on-disk record of (image, flange pose, detection, board pose) tuples,
plus the end-to-end solve+validate that turns a session into a calibration YAML.

Capture and solve are deliberately split: you capture once (with the robot), then can re-solve
offline as many times as you like (drop outliers, swap solver, tighten gates) with zero hardware.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import se3
from .config import DiversityGates, PassBars
from .diversity import assess_diversity
from .handeye import axxb_residuals, cross_solver_spread, solve_eye_to_hand, solve_robot_world
from .validate import base_frame_corner_error_m, leave_one_out, reject_outliers

__all__ = ["Capture", "CalibrationSession", "solve_session"]


@dataclass
class Capture:
    index: int
    image_path: str | None
    flange_pose7: list                 # [x,y,z,qw,qx,qy,qz]
    n_corners: int
    pnp_rms_px: float | None
    T_cam_board: list | None           # 4x4 as nested list
    laplacian_var: float | None = None

    @property
    def T_base_flange(self) -> np.ndarray:
        return se3.pose7_to_T(np.asarray(self.flange_pose7, dtype=float))

    @property
    def T_cam_board_mat(self) -> np.ndarray:
        return np.asarray(self.T_cam_board, dtype=float)


@dataclass
class CalibrationSession:
    root: str
    board: dict = field(default_factory=dict)
    zed_serial: str | None = None
    factory_K: list | None = None
    image_size: list | None = None
    captures: list = field(default_factory=list)

    @property
    def dir(self) -> Path:
        return Path(self.root)

    def add(self, cap: Capture):
        self.captures.append(cap)

    def usable(self):
        return [c for c in self.captures if c.T_cam_board is not None]

    def save(self) -> str:
        self.dir.mkdir(parents=True, exist_ok=True)
        doc = {
            "board": self.board,
            "zed_serial": self.zed_serial,
            "factory_K": self.factory_K,
            "image_size": self.image_size,
            "captures": [asdict(c) for c in self.captures],
        }
        p = self.dir / "session.json"
        p.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return str(p)

    @classmethod
    def load(cls, root: str | Path) -> CalibrationSession:
        root = Path(root)
        doc = json.loads((root / "session.json").read_text(encoding="utf-8"))
        s = cls(root=str(root), board=doc.get("board", {}), zed_serial=doc.get("zed_serial"),
                factory_K=doc.get("factory_K"), image_size=doc.get("image_size"))
        s.captures = [Capture(**c) for c in doc.get("captures", [])]
        return s


def solve_session(session: CalibrationSession, board_object_points,
                  gates: DiversityGates | None = None, bars: PassBars | None = None,
                  drop_outliers: bool = True) -> dict:
    """Full pipeline: diversity gate -> 5-solver hand-eye -> cross-checks -> validation -> verdict.

    ``board_object_points`` are the board's interior-corner coordinates in the board frame (metres),
    e.g. ``board.getChessboardCorners()``. Returns a report dict including ``T_base_camera`` (primary)
    and a top-level ``verdict`` in {PASS, WARN, FAIL}.
    """
    bars = bars or PassBars()
    usable = session.usable()
    flange = [c.T_base_flange for c in usable]
    boards = [c.T_cam_board_mat for c in usable]
    rms = [c.pnp_rms_px for c in usable]

    report: dict = {"n_captures": len(session.captures), "n_usable": len(usable)}

    div = assess_diversity(flange, gates)
    report["diversity"] = div.as_dict()
    if div.verdict == "FAIL":
        report["verdict"] = "FAIL"
        report["reason"] = "pose set is degenerate (see diversity); refusing to solve"
        return report

    # first solve (all usable) to seed outlier rejection
    first = solve_eye_to_hand(flange, boards)
    if drop_outliers and len(usable) >= max(6, (gates or DiversityGates()).min_poses):
        keep, dropped = reject_outliers(
            flange, boards, rms, T_base_camera=first.T_primary,
            rms_px_max=bars.per_frame_pnp_px_fail, axxb_mm_max=bars.axxb_translation_mm_fail)
        report["dropped"] = [{"index": i, "reason": r} for i, r in dropped]
        if len(keep) >= (gates or DiversityGates()).min_poses:
            flange = [flange[i] for i in keep]
            boards = [boards[i] for i in keep]
            rms = [rms[i] for i in keep]

    res = solve_eye_to_hand(flange, boards)
    T = res.T_primary
    report["T_base_camera"] = T.tolist()
    report["primary_solver"] = res.primary
    report["per_solver"] = {m: M.tolist() for m, M in res.T_base_camera.items()}

    sp_mm, sp_deg = cross_solver_spread(res.T_base_camera)
    report["cross_solver_spread"] = {"translation_mm": round(sp_mm, 3), "rotation_deg": round(sp_deg, 4)}

    try:
        T_rw, T_fb = solve_robot_world(flange, boards)
        d_mm = float(np.linalg.norm(T_rw[:3, 3] - T[:3, 3]) * 1000.0)
        d_deg = se3.rotation_angle_deg(T_rw[:3, :3] @ T[:3, :3].T)
        report["robot_world_crosscheck"] = {"translation_mm": round(d_mm, 3),
                                            "rotation_deg": round(d_deg, 4)}
    except Exception as e:
        report["robot_world_crosscheck"] = {"error": str(e)}

    ax = axxb_residuals(T, flange, boards)
    report["axxb"] = {k: round(v, 3) for k, v in ax.items() if k != "T_flange_board"}

    if board_object_points is not None:
        report["base_frame_corner_error"] = {
            k: round(v, 3) for k, v in
            base_frame_corner_error_m(T, flange, boards, board_object_points).items()}

    if len(flange) >= max(5, (gates or DiversityGates()).min_poses - 1):
        try:
            report["leave_one_out"] = {k: round(v, 3) if isinstance(v, float) else v
                                       for k, v in leave_one_out(flange, boards).items()}
        except Exception as e:
            report["leave_one_out"] = {"error": str(e)}

    report["verdict"] = _grade(report, div, bars)
    return report


def _grade(report, div, bars: PassBars) -> str:
    verdict = "PASS" if div.verdict != "WARN" else "WARN"
    sp = report.get("cross_solver_spread", {})
    if sp.get("translation_mm", 0) > bars.cross_solver_translation_mm_fail:
        return "FAIL"
    if sp.get("translation_mm", 0) > bars.cross_solver_translation_mm_pass:
        verdict = "WARN"
    if sp.get("rotation_deg", 0) > bars.cross_solver_rotation_deg_fail:
        return "FAIL"
    ax = report.get("axxb", {})
    if ax.get("translation_mm_max", 0) > bars.axxb_translation_mm_fail:
        verdict = "WARN"
    loo = report.get("leave_one_out", {})
    if isinstance(loo, dict) and loo.get("origin_std_mm", 0) > bars.loo_translation_std_mm_fail:
        verdict = "WARN"
    return verdict
