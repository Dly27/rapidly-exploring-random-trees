import numpy as np

from rrt.grids import grids


def test_random_cluttered_map(monkeypatch):
    monkeypatch.setattr(np.random, "randint", lambda *args: 1)
    expected_grid = np.array([[0, 0], [0, 1]])
    assert np.array_equal(grids.random_cluttered_map(size=2, n_rects=1), expected_grid)


def test_load_map(tmp_path):
    map_file = tmp_path / "test.map"

    map_file.write_text("type octile\nheight 2\nwidth 3\nmap\n.@T\nW%.\n")

    result = grids.load_map(map_file)

    expected = np.array(
        [
            [0, 1, 1],
            [1, 1, 0],
        ],
        dtype=np.uint8,
    )

    assert np.array_equal(result, expected)
