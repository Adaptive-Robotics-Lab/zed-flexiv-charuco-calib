"""Detection on a clean synthetic render of the board (needs cv2)."""
import numpy as np
import pytest

from zfcc.config import BoardConfig

cv2 = pytest.importorskip("cv2")
from zfcc.board import make_board, make_detector, n_interior_corners
from zfcc.detect import corner_coverage_bins, detect_charuco, laplacian_var


def _render(board, px=80, margin=40):
    cols, rows = board.getChessboardSize()
    return board.generateImage((cols * px + 2 * margin, rows * px + 2 * margin), marginSize=margin)


def test_detects_all_interior_corners_on_clean_render():
    board = make_board(BoardConfig())
    detector = make_detector(board, subpix=True)
    img = _render(board)
    det = detect_charuco(img, board, detector)
    assert det.ok
    # a clean fronto-parallel render should yield (nearly) every interior corner
    assert det.n_corners >= n_interior_corners(board) - 2


def test_coverage_bins_sum_to_corner_count():
    board = make_board(BoardConfig())
    detector = make_detector(board)
    img = _render(board)
    det = detect_charuco(img, board, detector)
    bins = corner_coverage_bins(det, (img.shape[1], img.shape[0]), grid=(3, 3))
    assert int(bins.sum()) == det.n_corners


def test_laplacian_var_drops_with_blur():
    board = make_board(BoardConfig())
    img = _render(board)
    sharp = laplacian_var(img)
    blurred = laplacian_var(cv2.GaussianBlur(img, (0, 0), 4))
    assert sharp > blurred > 0
