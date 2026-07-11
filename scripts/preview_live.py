#!/usr/bin/env python3
"""Live ZED viewfinder for camera placement, with ChArUco detection overlay.

Streams the SAME rectified-left stream the calibration consumes (zfcc.zed_io.ZedCamera,
factory K, self-calib disabled) as MJPEG over HTTP, so it works headless and can be
watched from any browser (including a phone next to the camera).

  http://<machine-ip>:<port>/        live view
  http://<machine-ip>:<port>/snap    save a full-res snapshot on the machine

Overlay:
  - detected ChArUco corners + count (green = board fully usable, yellow = partial)
  - PnP RMS (px) and board distance (m) whenever the pose solves
  - focus measure (Laplacian variance; higher = sharper)
  - 3x3 grid + centre crosshair (the intrinsics coverage gate bins corners 3x3)
  - FPS
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from zfcc.board import make_board, make_detector  # noqa: E402
from zfcc.config import BoardConfig, ZedConfig  # noqa: E402
from zfcc.detect import board_pose_pnp, detect_charuco  # noqa: E402
from zfcc.zed_io import ZedCamera  # noqa: E402

_latest_jpeg: bytes | None = None
_latest_raw: np.ndarray | None = None
_lock = threading.Lock()
_snap_dir = REPO / "runs" / "preview_snaps"

PAGE = b"""<!doctype html><title>ZED preview</title>
<body style="margin:0;background:#111;color:#eee;font-family:sans-serif">
<div style="padding:6px">ZED live preview &mdash; <a style="color:#8cf" href="/snap">save snapshot</a></div>
<img src="/stream" style="width:100%;max-width:1280px;display:block"/></body>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):  # quiet
        pass

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(PAGE)
        elif self.path == "/snap":
            with _lock:
                raw = None if _latest_raw is None else _latest_raw.copy()
            if raw is None:
                self.send_error(503, "no frame yet")
                return
            _snap_dir.mkdir(parents=True, exist_ok=True)
            p = _snap_dir / f"snap_{time.strftime('%H%M%S')}.png"
            cv2.imwrite(str(p), raw)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(f"saved {p}\n".encode())
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with _lock:
                        buf = _latest_jpeg
                    if buf is not None:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(buf)}\r\n\r\n".encode()
                                         + buf + b"\r\n")
                    time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


def capture_loop(a) -> None:
    global _latest_jpeg, _latest_raw
    bcfg = BoardConfig.load(a.board)
    board = make_board(bcfg)
    detector = make_detector(board)
    total_corners = (bcfg.squares_xy[0] - 1) * (bcfg.squares_xy[1] - 1)

    zcfg = ZedConfig.load(a.zed)
    if a.depth_mode:
        # The viewfinder only needs the rectified left image; NONE also sidesteps
        # a missing-TensorRT NEURAL failure without touching the calib config.
        zcfg.depth_mode = a.depth_mode
    with ZedCamera(zcfg) as cam:
        w, h = cam.image_size
        print(f"ZED {cam.serial} open: {w}x{h}; board expects {total_corners} inner corners")
        t_prev, fps = time.time(), 0.0
        while True:
            frame = cam.grab()
            img = frame.bgr.copy()
            det = detect_charuco(frame.gray, board, detector)

            for k in (1, 2):
                cv2.line(img, (w * k // 3, 0), (w * k // 3, h), (90, 90, 90), 1)
                cv2.line(img, (0, h * k // 3), (w, h * k // 3), (90, 90, 90), 1)
            cv2.drawMarker(img, (w // 2, h // 2), (200, 200, 200),
                           cv2.MARKER_CROSS, 24, 1)

            dist_txt, rms_txt = "-", "-"
            colour = (0, 200, 255)
            if det.ok:
                cv2.aruco.drawDetectedCornersCharuco(
                    img, det.charuco_corners, det.charuco_ids, (0, 255, 0))
                try:
                    T_cb, rms = board_pose_pnp(det, board, frame.K, D=None)
                    dist_txt = f"{float(np.linalg.norm(T_cb[:3, 3])):.3f} m"
                    rms_txt = f"{rms:.2f} px"
                    if det.n_corners >= int(0.8 * total_corners) and rms < 1.0:
                        colour = (0, 255, 0)
                except Exception:
                    pass

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now
            hud = (f"corners {det.n_corners}/{total_corners}   rms {rms_txt}   "
                   f"dist {dist_txt}   focus {det.laplacian_var:.0f}   fps {fps:.1f}")
            cv2.rectangle(img, (0, 0), (w, 34), (0, 0, 0), -1)
            cv2.putText(img, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 2)

            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                with _lock:
                    _latest_jpeg = buf.tobytes()
                    _latest_raw = frame.bgr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--board", default=str(REPO / "configs/board_calibio_9x14.yaml"))
    ap.add_argument("--zed", default=str(REPO / "configs/zed_2i_hd720.yaml"))
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--depth-mode", default="NONE",
                    help="override the config's depth_mode for preview ('' = keep config)")
    a = ap.parse_args()

    t = threading.Thread(target=capture_loop, args=(a,), daemon=True)
    t.start()
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    print(f"live view: http://0.0.0.0:{a.port}/  (Ctrl-C to stop)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
