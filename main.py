# Paper: https://msl.cs.illinois.edu/~lavalle/papers/Lav98c.pdf
# Paper: https://arxiv.org/pdf/1105.1186
# Paper: https://www.researchgate.net/publication/2539782_The_Bridge_Test_for_Sampling_Narrow_Passages_with_Probabilistic_Roadmap_Planners
# Maps: https://movingai.com/benchmarks/grids.html

from grids.grids import *
import numpy as np
from plotter import Plotter
from benchmarker import Benchmark

if __name__ == "__main__":
    grid_map = load_map("grids\street-map\London_1_512.map")
    start_coords = [1, 1]
    goal_coords = np.array([376, 240])
    rebuild_freq = 500
    step = 10
    sampler_method = "goal_biased"
    goal_bias = 0.05
    iterations = 50
    k = 5000
    r = 3
    grid_map[378][378] = 0  # Make sure goal is not in an obstacle
    test_iterations = 10
    sample_iterations = 50

    """
    benchmarker = Benchmark(grid_map=grid_map, x_init=start_coords, goal=goal_coords,
                            step=step, rebuild_freq=rebuild_freq, k=k, r=r, informed=True)

    benchmarker.test(sampler_method=sampler_method,
                     goal_bias=goal_bias,
                     test_iterations=test_iterations,
                     sample_iterations=sample_iterations)

    """
    plot = Plotter(grid_map=grid_map, x_init=start_coords, goal=goal_coords,
                   step=step, rebuild_freq=rebuild_freq, k=k, r=r,
                   sampler_method=sampler_method, goal_bias=goal_bias, iterations=iterations,
                   smooth=True, informed=True
                   )

    plot.plot_grid()
    
    plot2 = Plotter(grid_map=grid_map, x_init=start_coords, goal=goal_coords,
                   step=step, rebuild_freq=rebuild_freq, k=k, r=r,
                   sampler_method=sampler_method, goal_bias=goal_bias, iterations=iterations,
                   smooth=True,
                   )

    plot2.plot_grid()