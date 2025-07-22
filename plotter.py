from grids.grids import *

import numpy as np
import matplotlib.pyplot as plt
from RRT import RRT
from matplotlib.collections import LineCollection

class Plotter:
    def __init__(self, grid_map, x_init, goal, step, rebuild_freq, k, r,
                 sampler_method, goal_bias, iterations, smooth=False):

        self.grid_map = grid_map
        self.x_init = x_init
        self.goal = goal
        self.step = step
        self.rebuild_freq = rebuild_freq
        self.k = k
        self.r = r
        self.sampler_method = sampler_method
        self.goal_bias = goal_bias
        self.iterations = iterations
        self.smooth = smooth


    def plot_grid(self):
        rrt = RRT(
            x_init=self.x_init,
            grid_map=self.grid_map,
            rebuild_freq=self.rebuild_freq,
            goal=self.goal,
            step=self.step
        )

        rrt.select_sampler(
            sampler_method=self.sampler_method,
            goal_bias=self.goal_bias,
            iterations=self.iterations
        )

        rrt.grow(r=self.r, k=self.k)
        nearest_to_goal = rrt.find_nearest_neighbour(self.goal)
        path = rrt.get_path(goal_node=nearest_to_goal)

        if self.smooth:
            path = rrt.smooth_path()

        fig, ax = plt.subplots()

        # Draw obstacles
        ax.imshow(self.grid_map, cmap='Greys', origin='lower')

        # Draw tree edges
        segments = []
        for node in rrt.nodes:
            if node.parent is not None:
                segments.append([node.states, node.parent.states])
        lc = LineCollection(segments, colors='lightblue', linewidths=0.5)
        ax.add_collection(lc)

        # Draw tree nodes as one scatter
        xs = [node.states[0] for node in rrt.nodes]
        ys = [node.states[1] for node in rrt.nodes]
        ax.scatter(xs, ys, s=5, color='gray', label='Tree Nodes')

        # Draw path
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], color='red', linewidth=2, label='Path')

        # Start and goal
        ax.scatter(*self.x_init, color='green', label='Start')
        ax.scatter(*self.goal, color='blue', label='Goal')

        ax.set_title(f"{self.sampler_method}")
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()
        plt.show()
