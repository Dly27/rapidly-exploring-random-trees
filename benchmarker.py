from RRT import RRT
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_rrt(params):
    """
    Runs RRT once
    :return: array: Relevent performance metrics
    """
    grid_map, x_init, goal, step, rebuild_freq, k, r, sampler_method, goal_bias, sample_iterations = params

    rrt = RRT(x_init=x_init, grid_map=grid_map, rebuild_freq=rebuild_freq, goal=goal, step=step)
    rrt.select_sampler(sampler_method=sampler_method, goal_bias=goal_bias, iterations=sample_iterations)

    grow_start = time.perf_counter()
    rrt.grow(k=k, r=r)
    grow_end = time.perf_counter()

    goal_reached = rrt.goal_reached
    node_count = rrt.node_count

    if goal_reached:
        nearest = rrt.find_nearest_neighbour(goal)
        rrt.get_path(goal_node=nearest)
        path_length = rrt.path_cost()
    else:
        path_length = float('inf')

    return [
        goal_reached,
        path_length,
        grow_end - grow_start,
        node_count
    ]


class Benchmark:
    def __init__(self, grid_map, x_init, goal, step, rebuild_freq, k, r):
        self.grid_map = grid_map
        self.x_init = x_init
        self.goal = goal
        self.step = step
        self.rebuild_freq = rebuild_freq
        self.k = k
        self.r = r
        self.results = []
        self.stats = None

    def test(self, sampler_method, goal_bias, sample_iterations, test_iterations):
        """
        Tests multiple RRTs in batches through parallelisation
        :return: dictionary: Stats abouth the performace of the RRTs
        """
        args = [
            [
                self.grid_map,
                self.x_init,
                self.goal,
                self.step,
                self.rebuild_freq,
                self.k,
                self.r,
                sampler_method,
                goal_bias,
                sample_iterations
            ]
            for _ in range(test_iterations)
        ]
        results = []

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(run_rrt, arg) for arg in args]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        successes, path_lengths, grow_times, node_counts = zip(*results)

        path_lengths = np.array(path_lengths)
        finite_paths = path_lengths[np.isfinite(path_lengths)]

        self.stats = {
            "success_rate": sum(successes) / test_iterations,
            "mean_path_length": float(np.mean(finite_paths)) if len(finite_paths) > 0 else float('inf'),
            "mean_grow_time": np.mean(grow_times),
            "mean_nodes": np.mean(node_counts)
        }

        print(self.stats)
