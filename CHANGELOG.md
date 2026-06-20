# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); the project uses semantic versioning.

## [0.1.0] - 2026-06-19

Initial public release.

### Added
- Eye-to-hand ChArUco calibration for a **fixed ZED 2i** observing a board on the **Flexiv Rizon
  flange**, recovering the metric `T_base_camera`.
- `se3` pure-numpy SE(3) core (the single place rigid-transform inversion lives), scalar-first
  quaternions matching Flexiv/RDK.
- Five hand-eye solvers (TSAI, PARK, HORAUD, ANDREFF, DANIILIDIS) with DANIILIDIS as primary, plus an
  independent `calibrateRobotWorldHandEye` cross-check.
- **ChArUco intrinsics** (`fx, fy, cx, cy`, distortion) with an **audit mode** that confirms the
  rectified ZED factory K without overwriting it, and a free-solve mode.
- **Pose-diversity / degeneracy gates** that refuse a near-coplanar pose set (the failure mode of the
  old 8-coplanar-marker calibration).
- Validation suite: cross-solver spread, AX=XB residuals, leave-one-out stability, base-frame corner
  error, and a physical **touch test**.
- Drop-in `T_base_zed2i.yaml` writer matching ActAhead's loader schema.
- Guarded hardware shims (`zed_io`, `robot_io`) so the package installs, tests, and solves offline
  without the ZED SDK or flexivrdk.
- Synthetic, hardware-free test suite (round-trips a known extrinsic to prove the inversion sign) and
  GitHub Actions CI on Python 3.9/3.11/3.12.
