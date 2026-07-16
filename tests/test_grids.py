import pytest
import numpy as np

import grids.grids
from grids import *

def test_random_cluttered_map(monkeypatch):
    monkeypatch.setattr(np.random, "randint", lambda *args: 1)
    expected_grid = np.array([[0, 0], [0, 1]])
    assert np.array_equal(grids.random_cluttered_map(size=2, n_rects=1), expected_grid)

def test_load_map(tmp_path):
    map_file = tmp_path / "test.map"

    map_file.write_text(
        "type octile\n"
        "height 2\n"
        "width 3\n"
        "map\n"
        ".@T\n"
        "W%.\n")

    result = grids.load_map(map_file)

    expected = np.array([
        [0, 1, 1],
        [1, 1, 0],
    ], dtype=np.uint8)

    assert np.array_equal(result, expected)