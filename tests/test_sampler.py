from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt

from rrt.grids.grids import load_map
from rrt.rrt import Node
from rrt.sampler import Sampler


@dataclass
class SampleRun:
    grid_map: NDArray[np.uint8]
    x_init: list[int]
    goal_coords: np.ndarray
    rebuild_freq: int
    step: int
    sampler_method: str
    goal_bias: float
    iterations: int
    k: int
    r: int
    test_iterations: int
    sample_iterations: int
    test_map: np.ndarray


@pytest.fixture
def sample_run() -> SampleRun:
    map_path = Path("rrt") / "grids" / "street-map" / "London_1_512.map"
    grid_map = load_map(map_path)
    grid_map[376, 240] = 0

    return SampleRun(
        grid_map=grid_map,
        x_init=[1, 1],
        goal_coords=np.array([376, 240]),
        rebuild_freq=500,
        step=10,
        sampler_method="uniform",
        goal_bias=0.05,
        iterations=50,
        k=5000,
        r=3,
        test_iterations=10,
        sample_iterations=50,
        test_map=load_map(map_path),
    )


@pytest.fixture
def sampler(sample_run: SampleRun):
    return Sampler(
        sampler_method="uniform",
        goal=sample_run.goal_coords,
        goal_bias=sample_run.goal_bias,
        height=len(sample_run.grid_map),
        width=len(sample_run.grid_map[0]),
        grid_map=sample_run.grid_map,
        iterations=50,
    )


def test_clamp(sampler: Sampler):
    point1 = (sampler.width + 0.6, sampler.height + 0.6)
    point2 = (-0.5, -0.5)
    x1, y1 = sampler.clamp(point1)
    x2, y2 = sampler.clamp(point2)

    assert (x1, y1) == (sampler.width, sampler.height)
    assert (x2, y2) == (0, 0)


def test_is_in_obstacle(sampler: Sampler):
    sampler.grid_map[100, 100] = 1
    point1 = (100.5, 100.5)

    assert sampler.is_in_obstacle(point1)


def test_distance_from_obstacle(sampler: Sampler):
    sampler.grid_map[100, 100] = 1
    sampler.grid_map[99, 100] = 1
    sampler.grid_map[100, 99] = 0
    sampler.grid_map[99, 99] = 0

    point1 = (100, 100)
    point2 = (99, 99)
    sampler.distance_map = distance_transform_edt(sampler.grid_map == 0)

    assert int(sampler.distance_from_obstacle(point1)) == 0
    assert int(sampler.distance_from_obstacle(point2)) == 1


def test_uniform(sampler: Sampler):
    point = sampler.uniform()

    # Check in bounds
    assert np.all(np.array([0, 0]) <= point)
    assert np.all(point <= np.array([sampler.width, sampler.height]))

    assert point.shape == (2,)


def test_goal_biased(sampler: Sampler, monkeypatch):
    monkeypatch.setattr(sampler, "goal_bias", 1)

    assert np.array_equal(sampler.goal_biased(), sampler.goal)


def test_obstacle_biased(sampler: Sampler, monkeypatch):
    sampler.grid_map[:3, :3] = 1
    sampler.grid_map[1, 1] = 0

    p1 = np.array([1.5, 1.5])
    sampler.distance_map = distance_transform_edt(sampler.grid_map == 0)

    monkeypatch.setattr(sampler, "uniform", lambda: p1)
    monkeypatch.setattr(np.random, "normal", lambda *args, **kwargs: np.array([1, 1]))

    assert not np.array_equal(sampler.obstacle_biased(), p1)


def test_bridge(sampler: Sampler, monkeypatch):
    sampler.grid_map[:3, :3] = 1
    sampler.grid_map[1, 1] = 0

    sampler.distance_map = distance_transform_edt(sampler.grid_map == 0)

    p1 = np.array([0.5, 0.5])
    offset = np.array([2, 2])
    p2 = p1 + offset
    midpoint = (p1 + p2) / 2

    monkeypatch.setattr(sampler, "uniform", lambda: p1)
    monkeypatch.setattr(np.random, "normal", lambda *args, **kwargs: offset)
    monkeypatch.setattr(sampler, "iterations", 1)

    test_midpoint = sampler.bridge()
    print(f"Test midpoint: {test_midpoint}, Expected midpoint: {midpoint}")
    assert np.allclose(test_midpoint, midpoint)


def test_far_from_obstacle(sampler: Sampler, monkeypatch):
    points = [np.array([1, 1]), np.array([2, 2]), np.array([3, 3])]
    distances = {(1, 1): 2, (2, 2): 5, (3, 3): 1.0}

    monkeypatch.setattr(sampler, "iterations", len(points))
    point_iter = iter(points)

    monkeypatch.setattr(sampler, "uniform", lambda: next(point_iter))
    monkeypatch.setattr(sampler, "distance_from_obstacle", lambda point: distances[tuple(point)])

    assert np.array_equal(sampler.far_from_obstacle(), np.array([2.0, 2.0]))


def test_halton(sampler: Sampler, monkeypatch):
    monkeypatch.setattr(sampler.halton_sampler, "random", lambda n: [[1, 1]])
    point = np.array([sampler.width, sampler.height])

    assert np.array_equal(point, sampler.halton())


def test_informed(sampler: Sampler, monkeypatch):
    start = Node(np.array([0, 0]))
    goal = Node(np.array([4, 0]))
    best_cost = 6

    monkeypatch.setattr(np.random, "normal", lambda *args, **kwargs: np.array([1.0, 0.0]))
    monkeypatch.setattr(np.random, "rand", lambda: 1)
    result = sampler.informed(start, goal, best_cost)
    expected = np.array([3.5, 0])  # Use formula

    assert np.allclose(result, expected)


def test_line_based(sampler: Sampler, monkeypatch):
    sampler.grid_map = np.zeros((5, 5), dtype=np.uint8)
    sampler.width = 5
    sampler.height = 5
    sampler.iterations = 2

    sampler.grid_map[0, 1:4] = 1
    points = iter([0, 0, 4, 0, 0, 2, 4, 2])
    monkeypatch.setattr(np.random, "uniform", lambda *args, **kwargs: next(points))

    expected_midpoint = np.array([2, 2])
    test_midpoint = sampler.line_based()

    assert np.array_equal(test_midpoint, expected_midpoint)


def test_sample(sampler: Sampler, monkeypatch):
    expected_point = np.array([1.5, 2.5])
    sampler.sampler_type = "uniform"
    monkeypatch.setattr(sampler, "uniform", lambda: expected_point)
    sampler.methods["uniform"] = sampler.uniform
    test_point = sampler.sample()

    assert np.array_equal(test_point, expected_point)
