"""ChArUco board factory + detector (OpenCV >= 4.7 objdetect API; verified on cv2 4.12.0).

The OpenCV 4.6 -> 4.7 break removed ``CharucoBoard_create`` / ``Dictionary_get``; this module uses
only the new ``cv2.aruco.CharucoBoard((cols, rows), ...)`` + ``cv2.aruco.CharucoDetector`` path.
"""
from __future__ import annotations

from .config import BoardConfig

__all__ = ["make_dictionary", "make_board", "make_detector", "resolve_legacy_pattern", "render_board_png"]


def make_dictionary(name: str):
    import cv2

    if not hasattr(cv2.aruco, name):
        raise ValueError(f"unknown ArUco dictionary {name!r}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def make_board(cfg: BoardConfig):
    """Build the ChArUco board. ``cfg.squares_xy`` is (cols, rows) == (squaresX, squaresY)."""
    import cv2

    cols, rows = int(cfg.squares_xy[0]), int(cfg.squares_xy[1])
    board = cv2.aruco.CharucoBoard(
        (cols, rows), float(cfg.square_length_m), float(cfg.marker_length_m),
        make_dictionary(cfg.aruco_dict),
    )
    if cfg.legacy_pattern is not None:
        board.setLegacyPattern(bool(cfg.legacy_pattern))
    return board


def make_detector(board, subpix: bool = True):
    """CharucoDetector with sub-pixel marker-corner refinement (recommended)."""
    import cv2

    charuco_params = cv2.aruco.CharucoParameters()
    detector_params = cv2.aruco.DetectorParameters()
    if subpix:
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.CharucoDetector(board, charuco_params, detector_params)


def n_interior_corners(board) -> int:
    cols, rows = board.getChessboardSize()
    return (cols - 1) * (rows - 1)


def resolve_legacy_pattern(board, gray) -> bool:
    """Determine the board's chessboard parity empirically on one capture.

    Pre-4.6 / some third-party generators flip the origin square for even row counts. A bought
    board may match either; pick the flag that detects MORE charuco corners. Returns the chosen flag
    and leaves the board configured with it.
    """
    import cv2

    def count(flag: bool) -> int:
        board.setLegacyPattern(flag)
        det = cv2.aruco.CharucoDetector(board)
        cc, ci, _, _ = det.detectBoard(gray)
        return 0 if ci is None else int(len(ci))

    n_false = count(False)
    n_true = count(True)
    chosen = n_true > n_false
    board.setLegacyPattern(chosen)
    return chosen


def render_board_png(board, out_path: str, px_per_square: int = 120, margin_px: int = 40) -> str:
    """Render a printable board image to verify against the physical Calib.io target."""
    import cv2

    cols, rows = board.getChessboardSize()
    img = board.generateImage((cols * px_per_square + 2 * margin_px,
                               rows * px_per_square + 2 * margin_px), marginSize=margin_px)
    cv2.imwrite(out_path, img)
    return out_path
