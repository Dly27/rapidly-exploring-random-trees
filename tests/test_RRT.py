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

def test_initialize(sample_run):
    rrt = RRT(
        x_init=sample_run.x_init,
        grid_map=sample_run.grid_map,
        rebuild_freq=sample_run.rebuild_freq,
        goal=sample_run.goal_coords,
        step=sample_run.step,
    )

    assert rrt.node_count == 1
    assert len(rrt.nodes) == 1
    assert not rrt.goal_reached

def test_when_norm_is_zero(sample_run):
    rrt = RRT(
        x_init=sample_run.x_init,
        grid_map=sample_run.grid_map,
        rebuild_freq=sample_run.rebuild_freq,
        goal=sample_run.goal_coords,
        step=sample_run.step,
    )
    test_node = rrt.nodes[-1]
    direction = rrt.select_control(test_node, test_node.states)

    assert direction.sum() == 0

def test_point_in_obstacle(sample_run):
    flag = False

    rrt = RRT(
        x_init=sample_run.x_init,
        grid_map=sample_run.grid_map,
        rebuild_freq=sample_run.rebuild_freq,
        goal=sample_run.goal_coords,
        step=sample_run.step,
    )

    obstacle_indices = np.argwhere(sample_run.grid_map == 1)
    obstacle_point = obstacle_indices[0]

    assert not rrt.valid(p1=obstacle_point)




