import pytest
from rrt.plotter import Plotter
import numpy as np


@pytest.fixture
def plotter():
    grid_map = np.zeros((20, 20), dtype=np.uint8)

    return Plotter(
        grid_map=grid_map,
        x_init=np.array([1, 1]),
        goal=np.array([15, 15]),
        step=1,
        rebuild_freq=10,
        k=50,
        r=1,
        limit=5,
        sampler_method="uniform",
        goal_bias=0.05,
        iterations=10,
        smooth=False,
        informed=False,
    )


def test_plotter_initialisation(plotter: Plotter):
    assert np.array_equal(plotter.x_init, np.array([1, 1]))
    assert np.array_equal(plotter.goal, np.array([15, 15]))
    assert plotter.sampler_method == "uniform"
    assert not plotter.smooth
    assert not plotter.informed

def test_plot_grid_runs(plotter: Plotter):
    plotter.plot_grid()
    plotter.close_plots()