# Versions & compatibility

## OpenCV — the 4.6 → 4.7 API break (this matters)

The ArUco/ChArUco API was rewritten in **OpenCV 4.7**. This repo targets the **new** `objdetect` API
only and requires `opencv-contrib-python >= 4.7` (verified on **4.12.0**).

| Removed in 4.7 | Replacement (used here) |
|---|---|
| `cv2.aruco.CharucoBoard_create(sx, sy, sl, ml, dict)` | `cv2.aruco.CharucoBoard((cols, rows), sl, ml, dict)` |
| `cv2.aruco.Dictionary_get(...)` | `cv2.aruco.getPredefinedDictionary(...)` |
| `cv2.aruco.detectMarkers` + `interpolateCornersCharuco` | `cv2.aruco.CharucoDetector(board).detectBoard(img)` |
| `cv2.aruco.calibrateCameraCharuco(...)` | `board.matchImagePoints(...)` → `cv2.calibrateCamera(...)` |
| `cv2.aruco.estimatePoseCharucoBoard(...)` | `board.matchImagePoints(...)` → `cv2.solvePnP(...)` |

Note the constructor takes `(cols, rows)` = `(squaresX, squaresY)`. A board catalogued as "9×14"
(9 rows × 14 cols) is configured as `squares_xy: [14, 9]`.

`setLegacyPattern(True/False)` toggles the pre-4.6 chessboard parity; some third-party boards need it.
Leave `legacy_pattern: null` in the config to auto-resolve it on the first capture (the flag that
detects more corners wins).

## ZED 2i / ZED SDK

- The **left** stream is rectified ⇒ distortion ≈ 0; the factory `K` is read from the SDK and is the
  model the depth engine uses. Intrinsics default to **audit** (confirm, don't overwrite).
- `depth_mode: NEURAL` (the `ULTRA` mode is deprecated in SDK 5.x).
- `camera_disable_self_calib: true` freezes `K` across opens for reproducibility.
- Capture uses the Python `pyzed` API, installed via the ZED SDK (not from PyPI).

## Flexiv RDK / flexiv-control

- Poses are `[x, y, z, qw, qx, qy, qz]`, metres + scalar-first unit quaternion.
- Calibration reads the **flange** pose (`RobotStates.flange_pose`), not `tcp_pose`.
- Two transports: direct `flexivrdk.Robot`, or the
  [`flexiv-control`](https://pypi.org/project/flexiv-control/) `serve` daemon over its socket.

## Python

Tested on CPython **3.9 / 3.11 / 3.12** in CI. The math core is pure NumPy; OpenCV is imported lazily
so the package imports (and the synthetic tests that don't need cv2) work without it.
