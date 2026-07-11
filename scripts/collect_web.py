#!/usr/bin/env python3
"""Browser-driven eye-to-hand capture: live ChArUco view + one-tap pose recording.

Same math and session format as ``zfcc-collect`` (detect_charuco -> board_pose_pnp ->
FlangePoseReader -> CalibrationSession), but the record trigger is a button in the
browser (phone-friendly, next to the robot) instead of ENTER in a terminal, and the
page shows the LIVE diversity gates so you know when the set is good *before* leaving
the robot. The flange pose comes from flange_pose_helper.py running in a flexivrdk env.

  http://<ip>:<port>/        live view + RECORD / UNDO + gate dashboard
  session.json + capture_XXX.png are written after every capture (crash-safe),
  drop-in compatible with zfcc-inspect / zfcc-solve.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from zfcc.board import make_board, make_detector, resolve_legacy_pattern  # noqa: E402
from zfcc.config import BoardConfig, RobotConfig, ZedConfig  # noqa: E402
from zfcc.detect import board_pose_pnp, detect_charuco  # noqa: E402
from zfcc.diversity import assess_diversity  # noqa: E402
from zfcc.session import CalibrationSession, Capture  # noqa: E402
from zfcc.zed_io import ZedCamera  # noqa: E402

_lock = threading.Lock()
_latest_jpeg: bytes | None = None
STATE: dict = {"n": 0, "usable": 0, "last": None, "diversity": None, "error": None}


class Rig:
    """Owns the camera, board, session, and the flange-pose helper subprocess."""

    def __init__(self, a):
        self.a = a
        self.bcfg = BoardConfig.load(a.board)
        self.board = make_board(self.bcfg)
        self.detector = make_detector(self.board)
        self.total = (self.bcfg.squares_xy[0] - 1) * (self.bcfg.squares_xy[1] - 1)
        self.sess = CalibrationSession(root=a.session, board=self.bcfg.__dict__.copy())
        self.sess.dir.mkdir(parents=True, exist_ok=True)
        rcfg = RobotConfig.load(a.robot)
        self.helper = subprocess.Popen(
            [a.helper_python, str(REPO / "scripts/flange_pose_helper.py"),
             rcfg.serial or rcfg.host],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        ready = self._helper_read()
        if not ready.get("ready"):
            raise SystemExit(f"flange helper failed: {ready.get('error')}")
        print("flange helper ready (read-only RDK connection)")
        self.cam = None
        self.frame = None

    def _span_gate(self, det) -> tuple[bool, str]:
        """Reject one-SIDED detections (they bias planar PnP rotation), not merely
        low counts: detected corners must span >= span_frac of the board's extent in
        BOTH board axes, and meet a moderate count floor."""
        if det.n_corners < self.a.min_corners:
            return False, (f"only {det.n_corners}/{self.total} corners "
                           f"(need >= {self.a.min_corners})")
        all_obj = np.asarray(self.board.getChessboardCorners(), dtype=float)
        ids = np.asarray(det.charuco_ids).reshape(-1)
        got = all_obj[ids]
        full = all_obj.max(axis=0) - all_obj.min(axis=0)
        span = got.max(axis=0) - got.min(axis=0)
        fx, fy = span[0] / full[0], span[1] / full[1]
        if fx < self.a.span_frac or fy < self.a.span_frac:
            return False, (f"one-sided view: corners span {fx:.0%} x {fy:.0%} of the "
                           f"board (need >= {self.a.span_frac:.0%} in both axes); "
                           f"bring the whole board roughly into frame")
        return True, ""

    def _helper_read(self) -> dict:
        """Read the next 'ZFCC {...}' protocol line, skipping flexivrdk log noise."""
        while True:
            line = self.helper.stdout.readline()
            if not line:
                raise RuntimeError("flange helper exited")
            if line.startswith("ZFCC "):
                return json.loads(line[5:])

    def flange_pose7(self) -> list:
        with _lock:
            self.helper.stdin.write("get\n")
            self.helper.stdin.flush()
            resp = self._helper_read()
        if "pose7" not in resp:
            raise RuntimeError(f"flange read failed: {resp}")
        return resp["pose7"]

    def record(self) -> dict:
        """Record one pose, but REFUSE captures that would poison the solve.

        Session s002 post-mortem: per-pose residual correlates -0.61 with corner
        count -- one-sided partial-board views give planar PnP a biased rotation
        (~1 deg), which dominated the AX=XB residual. So: (1) require a nearly
        full board (>= min_corners); (2) require the arm to be SETTLED (board
        pose stable across two frames 0.25 s apart).
        """
        frame = self.frame
        if frame is None:
            raise RuntimeError("no frame yet")
        det = detect_charuco(frame.gray, self.board, self.detector)
        ok, why = self._span_gate(det)
        if not ok:
            return {"rejected": why}
        T1, rms1 = board_pose_pnp(det, self.board, frame.K, D=None)
        time.sleep(0.25)
        frame = self.frame
        det = detect_charuco(frame.gray, self.board, self.detector)
        ok, why = self._span_gate(det)
        if not ok:
            return {"rejected": f"settle check: {why}"}
        T_cb, rms = None, None
        try:
            T, rms = board_pose_pnp(det, self.board, frame.K, D=None)
            dt_mm = float(np.linalg.norm(T[:3, 3] - T1[:3, 3])) * 1000.0
            dR = T1[:3, :3] @ T[:3, :3].T
            ang = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
            if dt_mm > self.a.settle_mm or ang > self.a.settle_deg:
                return {"rejected": f"NOT SETTLED: board moved {dt_mm:.2f} mm / "
                                    f"{ang:.3f} deg between frames; let go and wait"}
            T_cb = T.tolist()
        except Exception as e:
            print(f"  PnP failed: {e}")
        pose7 = self.flange_pose7()
        idx = len(self.sess.captures)
        img_path = str(self.sess.dir / f"capture_{idx:03d}.png")
        cv2.imwrite(img_path, frame.bgr)
        self.sess.add(Capture(index=idx, image_path=img_path, flange_pose7=pose7,
                              n_corners=det.n_corners,
                              pnp_rms_px=(None if rms is None else float(rms)),
                              T_cam_board=T_cb, laplacian_var=det.laplacian_var))
        self.sess.save()
        info = {"index": idx, "corners": det.n_corners, "total": self.total,
                "rms": None if rms is None else round(float(rms), 3),
                "usable": T_cb is not None,
                "dist": None if T_cb is None else round(float(np.linalg.norm(np.asarray(T_cb)[:3, 3])), 3)}
        print(f"  pose#{idx}: {info}")
        self._refresh_state(last=info)
        return info

    def undo(self) -> dict:
        if not self.sess.captures:
            return {"n": 0}
        cap = self.sess.captures.pop()
        if cap.image_path and Path(cap.image_path).exists():
            Path(cap.image_path).unlink()
        self.sess.save()
        self._refresh_state(last={"undone": cap.index})
        return {"undone": cap.index}

    def _refresh_state(self, last=None):
        usable = self.sess.usable()
        div = None
        if len(usable) >= 3:
            rep = assess_diversity([c.T_base_flange for c in usable],
                                   board_poses_T=[c.T_cam_board_mat for c in usable])
            div = rep.as_dict()
        STATE.update({"n": len(self.sess.captures), "usable": len(usable),
                      "diversity": div, "last": last or STATE.get("last")})


PAGE = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>zfcc capture</title>
<body style="margin:0;background:#111;color:#eee;font-family:sans-serif">
<img src="/stream" style="width:100%;display:block"/>
<div style="padding:10px">
<button onclick="rec()" style="font-size:28px;padding:14px 40px;background:#2a2;color:#fff;border:0;border-radius:10px">RECORD</button>
<button onclick="undo()" style="font-size:18px;padding:14px 20px;margin-left:12px;background:#a33;color:#fff;border:0;border-radius:10px">undo last</button>
<pre id=s style="font-size:15px;white-space:pre-wrap"></pre></div>
<script>
async function rec(){const r=await fetch('/record',{method:'POST'});document.getElementById('s').textContent='record: '+await r.text()+'\\n'+document.getElementById('s').textContent;}
async function undo(){await fetch('/undo',{method:'POST'});}
setInterval(async()=>{const r=await fetch('/state');const j=await r.json();
let d=j.diversity?JSON.stringify(j.diversity,null,1):'(need 3+ usable poses)';
document.getElementById('s').textContent='captures: '+j.n+'  usable: '+j.usable+'\\nlast: '+JSON.stringify(j.last)+'\\ndiversity: '+d;},1500);
</script></body>"""


def make_handler(rig: Rig):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, code, ctype, body: bytes):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self._send(200, "text/html", PAGE.encode())
            elif self.path == "/state":
                self._send(200, "application/json", json.dumps(STATE).encode())
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        buf = _latest_jpeg
                        if buf is not None:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                             + f"Content-Length: {len(buf)}\r\n\r\n".encode()
                                             + buf + b"\r\n")
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

        def do_POST(self):
            try:
                if self.path == "/record":
                    self._send(200, "application/json", json.dumps(rig.record()).encode())
                elif self.path == "/undo":
                    self._send(200, "application/json", json.dumps(rig.undo()).encode())
                else:
                    self.send_error(404)
            except Exception as e:
                self._send(500, "text/plain", str(e).encode())

    return Handler


def capture_loop(rig: Rig, a) -> None:
    global _latest_jpeg
    zcfg = ZedConfig.load(a.zed)
    if a.depth_mode:
        zcfg.depth_mode = a.depth_mode
    with ZedCamera(zcfg) as cam:
        rig.sess.zed_serial = cam.serial
        rig.sess.factory_K = cam.K.tolist()
        rig.sess.image_size = list(cam.image_size)
        if rig.bcfg.legacy_pattern is None:
            lp = resolve_legacy_pattern(rig.board, cam.grab().gray)
            print(f"resolved legacy_pattern={lp}")
        w, h = cam.image_size
        print(f"ZED {cam.serial}: {w}x{h}; board {rig.total} corners; session {rig.sess.dir}")
        while True:
            frame = cam.grab()
            rig.frame = frame
            img = frame.bgr.copy()
            det = detect_charuco(frame.gray, rig.board, rig.detector)
            colour = (0, 200, 255)
            rms_txt, dist_txt = "-", "-"
            if det.ok:
                cv2.aruco.drawDetectedCornersCharuco(
                    img, det.charuco_corners, det.charuco_ids, (0, 255, 0))
                try:
                    T_cb, rms = board_pose_pnp(det, rig.board, frame.K, D=None)
                    dist_txt = f"{float(np.linalg.norm(T_cb[:3, 3])):.3f}m"
                    rms_txt = f"{rms:.2f}px"
                    if det.n_corners >= int(0.8 * rig.total) and rms < 1.0:
                        colour = (0, 255, 0)
                except Exception:
                    pass
            hud = (f"[{STATE['usable']}/{STATE['n']} poses] corners {det.n_corners}/{rig.total} "
                   f"rms {rms_txt} dist {dist_txt} focus {det.laplacian_var:.0f}")
            cv2.rectangle(img, (0, 0), (w, 34), (0, 0, 0), -1)
            cv2.putText(img, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 2)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                _latest_jpeg = buf.tobytes()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", required=True)
    ap.add_argument("--board", default=str(REPO / "configs/board_calibio_9x14.yaml"))
    ap.add_argument("--zed", default=str(REPO / "configs/zed_2i_hd720.yaml"))
    ap.add_argument("--robot", default=str(REPO / "configs/rizon4s.yaml"))
    ap.add_argument("--helper-python",
                    default="/home/lab/miniconda3/envs/flexiv_env/bin/python")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--depth-mode", default="NONE")
    ap.add_argument("--min-corners", type=int, default=70,
                    help="moderate corner-count floor (the real gate is --span-frac)")
    ap.add_argument("--span-frac", type=float, default=0.75,
                    help="detected corners must span this fraction of the board in "
                         "both axes (one-sided planar PnP is rotation-biased; s002 lesson)")
    ap.add_argument("--settle-mm", type=float, default=0.5)
    ap.add_argument("--settle-deg", type=float, default=0.1)
    a = ap.parse_args()

    rig = Rig(a)
    t = threading.Thread(target=capture_loop, args=(rig, a), daemon=True)
    t.start()
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(rig))
    print(f"capture UI: http://0.0.0.0:{a.port}/")
    try:
        srv.serve_forever()
    finally:
        rig.helper.stdin.write("quit\n")


if __name__ == "__main__":
    main()
