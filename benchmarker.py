from RRT import RRT
import numpy as np
import time

class Benchmark:
    def __init__(self, grid_map, x_init, goal, step, rebuild_freq, k, r):
        self.grid_map = grid_map
        self.x_init = x_init
        self.goal = goal
        self.step = step
        self.rebuild_freq = rebuild_freq
        self.results = []
        self.k = k
        self.r = r

    def run_test(self, sampler_method, goal_bias, sample_iterations, run_iterations):
        success_count = 0
        path_lengths = []
        grow_times = []
        build_times = []
        node_counts =[]

        for _ in range(run_iterations):
            start = time.time()
            rrt = RRT(x_init=self.x_init, grid_map=self.grid_map, rebuild_freq=self.rebuild_freq,
                      goal=self.goal, step=self.step)
            end = time.time()
            build_times.append(end - start)

            rrt.select_sampler(sampler_method=sampler_method, goal_bias=goal_bias, iterations=sample_iterations)

            start = time.time()
            rrt.grow(k=self.k, r=self.r)
            end = time.time()
            grow_times.append(end - start)

            if rrt.goal_reached == True:
                success_count += 1

            node_counts.append(rrt.node_count)
            nearest_to_goal = rrt.find_nearest_neighbour(self.goal)
            rrt.get_path(goal_node=nearest_to_goal)
            path_lengths.append(rrt.path_cost())

        stats = {
            "success_rate": success_count / run_iterations,
            "mean_path_length": sum(path_lengths) / run_iterations,
            "mean_grow_time": sum(grow_times) / run_iterations,
            "mean_build_time": sum(build_times) / run_iterations,
            "mean_nodes": sum(node_counts) / run_iterations
        }

        return stats
