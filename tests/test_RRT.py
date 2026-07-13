import pytest
from dataclasses import dataclass
from RRT import RRT
from grids.grids import *
import numpy as np
from numpy.typing import NDArray

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
    map_path = r"grids\street-map\London_1_512.map"

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
    flag = False
    obstacle_indices = np.argwhere(rrt.grid_map == 1)
    obstacle_point = obstacle_indices[0]

    assert not rrt.valid(p1=obstacle_point)

def test_obstacle_between_nodes(rrt: RRT):
    grid = np.ones((100,100))
    grid[0,50] = 0
    grid[99,50] = 0
    rrt.grid_map = grid

    assert not rrt.valid([0,50], [100,50])

def test_no_goal_point(rrt: RRT):
    rrt.goal = None

    with pytest.raises(ValueError):
        rrt.grow(k=5000, r=3)

def test_sampler_not_selected(rrt: RRT):
    with pytest.raises(RuntimeError):
        rrt.grow(k=5000, r=3)

def test_zero_grow_iterations(rrt: RRT):
    rrt.select_sampler()
    rrt.grow(k=0, r =3)

    assert rrt.node_count == 1

def test_first_path_not_found_before_limit(rrt:RRT):
    rrt.select_sampler()

    with pytest.raises(ValueError):
        rrt.informed_grow(k=1000, r=3, limit=1)
