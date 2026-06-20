# zed-flexiv-charuco-calib

**Strict eye-to-hand calibration between a fixed ZED 2i camera and a Flexiv Rizon arm, using a
ChArUco board on the flange — with a falsifiable validation suite.**

[![ci](https://github.com/ZihaoLu001/zed-flexiv-charuco-calib/actions/workflows/ci.yml/badge.svg)](https://github.com/ZihaoLu001/zed-flexiv-charuco-calib/actions/workflows/ci.yml)
&nbsp;Apache-2.0&nbsp;·&nbsp;Python 3.9–3.12&nbsp;·&nbsp;OpenCV ≥ 4.7 (`opencv-contrib`)

It recovers the fixed rigid transform **`T_base_camera`** (camera pose in the robot base frame) so
that a 3D point the camera sees can be commanded directly in robot coordinates — the number a grasp
pipeline depends on. The output is a drop-in `T_base_zed2i.yaml`.

> Built to replace an ad-hoc calibration from **8 coplanar ArUco markers on the table**. That layout
> is *planar*: PnP is accurate in the image plane but weak in depth/tilt, so the recovered camera
> pose can be off by **1–2 cm in Z** while every on-screen check still looks perfect. This repo makes
> that failure mode **impossible to ship silently** — a pose-diversity gate refuses a near-coplanar
> set, and a five-solver cross-check plus a physical touch test must agree before a calibration is
> written.

---

## Does a ChArUco board also give camera intrinsics? — Yes.

A ChArUco board recovers the **full pinhole model** (`fx, fy, cx, cy` + distortion `k1,k2,p1,p2,k3`),
exactly like a plain chessboard but more robust to partial/occluded views because every detected
corner carries a unique ID. So one board does **both** jobs:

| Quantity | How | Notes |
|---|---|---|
| **Intrinsics** `K`, distortion | `board.matchImagePoints(...)` → `cv2.calibrateCamera(...)` | `cv2.aruco.calibrateCameraCharuco` was removed in OpenCV 4.7; this is the current path |
| **Board pose** `T_cam_board` | `board.matchImagePoints(...)` → `cv2.solvePnP(IPPE)` + LM refine | the per-view measurement that feeds hand-eye |
| **Extrinsics** `T_base_camera` | five `cv2.calibrateHandEye` solvers + `calibrateRobotWorldHandEye` cross-check | the goal of this repo |

**Important for the ZED 2i specifically:** the SDK's *left* image is already **rectified** against a
factory-calibrated `K` that the **depth engine also uses**. Re-deriving and overwriting `K` would
desync RGB and depth. So intrinsics here default to **audit mode**: seed the factory `K`, fix focal
length + principal point + zero distortion, and confirm it reprojects the board with sub-pixel RMS;
a free solve is also reported for transparency. Use `--mode free` only if you deliberately want an
independent `K` (e.g. for a non-ZED camera or a raw/unrectified stream).

---

## The method (eye-to-hand), in one paragraph

The camera is **fixed** in the workspace; the ChArUco board is **bolted to the flange**. For each
robot pose we read the **flange** pose `T_base_flange` (from the Flexiv controller) and measure the
board pose `T_cam_board` (from the image). Two unknown rigid transforms are constant across all
poses: the camera in base, `T_base_camera`, and the board on the flange, `T_flange_board`. They obey

```
T_base_camera · T_cam_board  =  T_base_flange · T_flange_board       (for every pose)
```

`cv2.calibrateHandEye` natively solves the *eye-in-hand* problem; for a **fixed camera** you feed the
**inverted** robot poses (`T_gripper_base = inv(T_base_flange)`) and the solver returns
`X = T_base_camera` directly. That single inversion is the whole eye-to-hand trick, and it lives in
exactly one place ([`handeye.solve_eye_to_hand`](src/zfcc/handeye.py)); a synthetic round-trip test
asserts the sign is right (a flipped inversion still returns a clean matrix — just the wrong one).

> This matches your description exactly: *"the ZED 2i estimates the 3D pose of the known-size ChArUco
> board from the image; those board poses are paired with the robot end-effector pose, and hand-eye
> calibration determines the fixed transform between the ZED 2i frame and the robot base frame."* The
> one refinement: pair board poses with the **flange** pose, not the **TCP** pose — the TCP carries
> the gripper tool offset, which would otherwise leak straight into the extrinsic.

See [docs/METHOD.md](docs/METHOD.md) for the full derivation and [docs/FRAMES.md](docs/FRAMES.md) for
frame/quaternion conventions.

---

## Why it is *strict* (the validation suite)

A solver always returns *a* matrix. These checks decide whether to trust it — and refuse to write a
calibration that fails. Every threshold lives in [`configs/pass_bars.yaml`](configs/pass_bars.yaml).

| Check | What it catches | Bar (default) |
|---|---|---|
| **Pose-diversity gate** | near-coplanar / single-axis pose sets (the old-calibration failure) | ≥3 rotation axes, coplanarity index ≥ 0.04 |
| **Five-solver agreement** | unstable geometry; one solver disagreeing | TSAI/PARK/HORAUD/ANDREFF/DANIILIDIS spread < 2 mm / 0.2° |
| **Robot-world cross-check** | a wholly independent algorithm should land on the same answer | `calibrateRobotWorldHandEye` within tolerance |
| **AX=XB residual** | per-pose inconsistency, bad pairings | max < 2 mm |
| **Per-frame PnP RMS** | blurry / grazing-angle views | drop views > 1 px |
| **Leave-one-out** | a single pose leveraging the fit | origin std < 5 mm |
| **Physical touch test** | the *whole* chain the robot uses — the metric that actually predicts grasps | < 3 mm at the tip |

On a synthetic 22-pose session with realistic detection noise (0.4 mm / 0.03°), the pipeline recovers
the ground-truth extrinsic to **~0.1 mm**, with five-solver spread **0.14 mm**, robot-world
cross-check **0.08 mm**, and leave-one-out origin std **0.04 mm** (reproduced in CI by
`tests/test_session.py` / `tests/test_handeye_synthetic.py`).

---

## Install

```bash
pip install -e .            # numpy, opencv-contrib-python, pyyaml
pip install -e ".[dev]"     # + pytest, ruff, pre-commit
```

Hardware capture additionally needs the **ZED SDK** (`pyzed`) and the Flexiv RDK (`flexivrdk`) or a
running [`flexiv-control`](https://pypi.org/project/flexiv-control/) `serve` daemon. Neither is
required to install the package, run the tests, or **solve from a saved session** — the hardware
shims use guarded imports.

## Quickstart

```bash
# 0. Verify the printed board matches your physical Calib.io target (counts, dict, square sizes!)
zfcc-render-board --board configs/board_calibio_9x14.yaml --out board.png

# 1. (optional) Audit the ZED factory intrinsics with ChArUco
zfcc-intrinsics  --zed configs/zed_2i_hd720.yaml --board configs/board_calibio_9x14.yaml --frames 20

# 2. Capture an eye-to-hand session: many DIVERSE flange poses, board always in view
zfcc-collect --session runs/s001 --board configs/board_calibio_9x14.yaml \
             --zed configs/zed_2i_hd720.yaml --robot configs/rizon4s.yaml --mode manual

# 3. Check coverage/diversity before you leave the robot
zfcc-inspect --session runs/s001

# 4. Solve offline + validate; writes T_base_zed2i.yaml only if the verdict isn't FAIL
zfcc-solve --session runs/s001 --board configs/board_calibio_9x14.yaml --out T_base_zed2i.yaml

# 5. Physically confirm the end-to-end accuracy (the number that predicts grasps)
zfcc-touch-test --calib T_base_zed2i.yaml --board configs/board_calibio_9x14.yaml \
                --zed configs/zed_2i_hd720.yaml --corner 0
```

Each console script is mirrored by a file in [`scripts/`](scripts/) (`python scripts/x.py …`).

## Output: drop-in `T_base_zed2i.yaml`

```yaml
T_base_zed2i:
  frame_convention: "T_A_B maps a point expressed in frame B into frame A: p_A = T_A_B @ p_B"
  parent_frame: flexiv_world
  child_frame: zed2i_camera_frame
  matrix: [[...4x4...]]            # base <- camera
  translation_xyz_m: [...]
  quaternion_wxyz: [...]           # scalar-first (Flexiv convention)
  inverse_T_camera_base: [[...]]
validation: { verdict: PASS, cross_solver_spread: {...}, axxb: {...}, leave_one_out: {...} }
```

Consume it as `p_base = T_base_zed2i.matrix @ [p_camera; 1]`.

## ⚠️ Before you trust any result

1. **Confirm board geometry.** A wrong `square_length_m` scales the whole extrinsic linearly. Measure
   your physical board and edit [`configs/board_calibio_9x14.yaml`](configs/board_calibio_9x14.yaml)
   (`squares_xy` is **(cols, rows)**; verify the ArUco dictionary too).
2. **Pair the FLANGE pose, not the TCP pose** (see method above).
3. **Make the poses diverse** — ≥20 poses, ≥3 non-parallel wrist-rotation axes, varied camera
   distance. The diversity gate will refuse a degenerate set; that is the point.
4. **Run the touch test.** A great reprojection RMS can still hide a bad chain.

## Repository layout

```
src/zfcc/        se3, board, detect, intrinsics, handeye, diversity, validate, touch_test,
                 yaml_out, session, zed_io, robot_io, _cli
scripts/         thin CLI wrappers over zfcc._cli
configs/         board / zed / robot / pass_bars YAML
tests/           synthetic, hardware-free suite (round-trips a known extrinsic)
docs/            METHOD, FRAMES, PROCEDURE, PASS_BARS, VERSIONS
```

## License

Apache-2.0 © 2026 Zihao Lu. See [LICENSE](LICENSE).
