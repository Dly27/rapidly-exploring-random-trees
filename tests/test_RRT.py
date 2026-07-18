from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from grids.grids import load_map
from rrt.RRT import RRT, Node


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
    map_path = Path("grids") / "street-map" / "London_1_512.map"
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
def rrt(sample_run: SampleRun) -> RRT:
    return RRT(
        x_init=sample_run.x_init,
        grid_map=sample_run.grid_map,
        rebuild_freq=sample_run.rebuild_freq,
        goal=sample_run.goal_coords,
        step=sample_run.step,
    )


def test_initialize(rrt: RRT):
    assert rrt.node_count == 1
    assert len(rrt.nodes) == 1
    assert not rrt.goal_reached


def test_when_norm_is_zero(rrt: RRT):
    test_node = rrt.nodes[-1]
    direction = rrt.select_control(test_node, test_node.states)

    assert direction.sum() == 0


def test_point_in_obstacle(rrt: RRT):
    obstacle_indices = np.argwhere(rrt.grid_map == 1)
    obstacle_point = obstacle_indices[0]

    assert not rrt.valid(p1=obstacle_point)


def test_obstacle_between_nodes(rrt: RRT):
    grid = np.ones((100, 100))
    grid[0, 50] = 0
    grid[99, 50] = 0
    rrt.grid_map = grid

    assert not rrt.valid([0, 50], [100, 50])


def test_sampler_not_selected(rrt: RRT):
    with pytest.raises(RuntimeError):
        rrt.grow(k=500, r=3)


def test_zero_grow_iterations(rrt: RRT):
    rrt.select_sampler()
    rrt.grow(k=0, r=3)

    assert rrt.node_count == 1


def test_first_path_not_found_before_limit(rrt: RRT):
    # Tests if the first path needed for informed grow is found
    rrt.select_sampler()

    with pytest.raises(RuntimeError):
        rrt.informed_grow(k=500, r=3, limit=1)


def test_informed_grow_updates_best_cost(rrt: RRT, monkeypatch):
    rrt.select_sampler()
    rrt.goal_reached = True
    goal_node = Node(rrt.goal)
    near_node = Node(np.array([0, 0]))

    monkeypatch.setattr(
        rrt,
        "find_nearest_neighbour",
        lambda point: goal_node if np.array_equal(point, rrt.goal) else near_node,
    )

    monkeypatch.setattr(rrt, "get_path", lambda node: None)
    costs = iter([100, 80])
    monkeypatch.setattr(rrt, "path_cost", lambda: next(costs))

    monkeypatch.setattr(
        rrt.sampler,
        "informed",
        lambda start, goal, best_cost: rrt.goal.copy(),
    )

    monkeypatch.setattr(
        rrt,
        "select_control",
        lambda x_near, x_random: np.array([1, 1]),
    )

    monkeypatch.setattr(
        rrt,
        "new_state",
        lambda x_near, u: rrt.goal.copy(),
    )

    monkeypatch.setattr(rrt, "valid", lambda p1, p2: True)
    monkeypatch.setattr(rrt, "add_state", lambda states, parent: None)

    rrt.informed_grow(k=1, r=3)

    assert rrt.best_cost < 100


def test_informed_grow_does_not_add_invalid_nodes(rrt: RRT, monkeypatch):
    rrt.select_sampler()
    rrt.goal_reached = True
    rrt.grid_map[50, 50] = 1
    test_node = [50, 50]
    initial_node_count = rrt.node_count

    monkeypatch.setattr(rrt.sampler, "informed", lambda start, goal, best_cost: test_node)

    monkeypatch.setattr(
        rrt,
        "select_control",
        lambda x_near, x_random: np.array([1, 1]),
    )
    monkeypatch.setattr(rrt, "new_state", lambda x_near, u: test_node)

    rrt.informed_grow(k=1, r=3)
    assert rrt.node_count == initial_node_count


def test_smooth_path_removes_unnecessary_nodes(rrt: RRT, monkeypatch):
    start = np.array([0, 0])
    point_1 = np.array([1, 1])
    point_2 = np.array([2, 2])
    goal = np.array([3, 3])

    rrt.path = [start, point_1, point_2, goal]

    monkeypatch.setattr(
        rrt,
        "valid",
        lambda p1, p2: True,
    )

    smoothed_path = rrt.smooth_path()

    assert np.array_equal(smoothed_path[0], start)
    assert np.array_equal(smoothed_path[1], goal)
