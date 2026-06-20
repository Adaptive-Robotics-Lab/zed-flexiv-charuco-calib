"""Board factory (needs cv2 for the aruco API)."""
import numpy as np
import pytest

from zfcc.config import BoardConfig

cv2 = pytest.importorskip("cv2")
from zfcc.board import make_board, make_dictionary, n_interior_corners, render_board_png


def test_board_dimensions_cols_rows():
    board = make_board(BoardConfig())  # squares_xy = (14, 9) -> (cols, rows)
    cols, rows = board.getChessboardSize()
    assert (cols, rows) == (14, 9)
    assert n_interior_corners(board) == (14 - 1) * (9 - 1) == 104


def test_unknown_dictionary_raises():
    with pytest.raises(ValueError):
        make_dictionary("DICT_NOPE_999")


def test_render_writes_image(tmp_path):
    board = make_board(BoardConfig())
    p = render_board_png(board, str(tmp_path / "b.png"), px_per_square=40)
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    assert img is not None and img.shape[0] > 0 and img.shape[1] > 0
