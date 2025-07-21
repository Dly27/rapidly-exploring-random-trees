# Paper: https://msl.cs.illinois.edu/~lavalle/papers/Lav98c.pdf
# Paper: https://arxiv.org/pdf/1105.1186
# Paper: https://www.researchgate.net/publication/2539782_The_Bridge_Test_for_Sampling_Narrow_Passages_with_Probabilistic_Roadmap_Planners
from grids.grids import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from scipy.stats.qmc import Halton
import time


def bresenham(x0, y0, x1, y1):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


class Sampler:
    def __init__(self, sampler_method, goal, goal_bias, height, width, grid_map, iterations=50):
        self.sampler_type = sampler_method
        self.goal = goal
        self.goal_bias = goal_bias
        self.height = height
        self.width = width
        self.grid_map = grid_map
        self.distance_map = distance_transform_edt((self.grid_map == 0))
        self.iterations = iterations
        self.halton_sampler = Halton(d=2, scramble=False)
        self.halton_index = 0

        self.methods = {
            "uniform": self.uniform,
            "goal_biased": self.goal_biased,
            "obstacle_biased": self.obstacle_biased,
            "bridge": self.bridge,
            "halton": self.halton,
            "far": self.far_from_obstacle,
            "line_based": self.line_based
        }

    def clamp(self, point):
        x, y = point
        x, y = int(x), int(y)

        # Clamp indices to valid range
        x = max(0, min(x, self.grid_map.shape[1] - 1))
        y = max(0, min(y, self.grid_map.shape[0] - 1))

        return x, y


    def is_in_obstacle(self, point):
        """
        Checks wheteher a given point is in an obstacle
        :param point: Point to check
        :return: True if in obstacle, False otherwise
        """
        x, y = self.clamp(point=point)
        return self.grid_map[y][x] == 1

    def distance_from_obstacle(self, point):
        x, y = self.clamp(point=point)
        return self.distance_map[y][x]

    def uniform(self):
        """
        Uniformly samples a point
        :return: (float, float): A 2D point in the grid map
        """
        return np.array([np.random.uniform(0, self.width), np.random.uniform(0, self.height)])

    def goal_biased(self):
        """
        Adds node towards the goal point with probability less than self.goal_bias,
        point is uniformly sampled otherwise
        :return: (float, float): A 2D point in the grid map
        """
        if np.random.rand() < self.goal_bias:
            return self.goal
        else:
            return self.uniform()

    def obstacle_biased(self, sigma=5):
        """
        Uniformly samples one point and use a Gaussian sample for another point located around the first
        and returns the point closest to an obstacle
        :return: (float, float): A 2D point in the grid map
        """
        p1 = self.uniform()
        p2 = np.clip(p1 + np.random.normal(0, sigma, size=2), [0, 0], [self.width, self.height])  # Clip to keep p2 in grid_map bounds
        if self.distance_from_obstacle(p1) <= self.distance_from_obstacle(p2):
            return p1
        else:
            return p2

    def bridge(self, sigma=5):
        """
        Uniformly samples self.iterations number of points and selects a midpoint of two points which are
        located in distinct obstacles, if found
        :return: (float, float): A 2D point in the grid map
        """
        for i in range(self.iterations):
            p1 = self.uniform()
            p2 = np.clip(p1 + np.random.normal(0, sigma, size=2), [0, 0], [self.width, self.height])  # Clip to keep p2 in grid_map bounds
            midpoint = (p1 + p2) / 2

            if self.is_in_obstacle(p1) and self.is_in_obstacle(p2) and not self.is_in_obstacle(midpoint):
                return midpoint
        return self.uniform()

    def far_from_obstacle(self):
        """
        Uniformly samples self.iterations number of points and selects the point farthest from its closest obstacle
        :return: (float, float): A 2D point in the grid map
        """
        points = [self.uniform() for _ in range(self.iterations)]
        distances = [self.distance_from_obstacle(point) for point in points]
        return points[np.argmax(distances)]

    def halton(self):
        """
        Returns next point in the Halton low-discrepency sequence scaled by the width and height of the grid map
        :return: (float, float): A 2D point in the grid map
        """
        point = self.halton_sampler.random(n=1)[0]
        self.halton_index += 1
        x = point[0] * self.width
        y = point[1] * self.height

        return np.array([x, y])

    def line_based(self):
        """
        Samples several lines and picks the midpoint of the line with the least obstacle intersections.
        :return: (float, float): A 2D point (midpoint) from the line intersecting the most obstacles.
        """
        min_intersections = float("inf")
        x0, y0, x1, y1 = None, None, None, None

        for _ in range(self.iterations):
            x2, y2 = np.random.uniform(0, self.width), np.random.uniform(0, self.height)
            x3, y3 = np.random.uniform(0, self.width), np.random.uniform(0, self.height)
            count = 0
            x2, y2, x3, y3 = int(x2), int(y2), int(x3), int(y3)

            for x, y in bresenham(x2, y2, x3, y3):
                if self.grid_map[y][x] == 1:
                    count += 1

            if count < min_intersections:
                min_intersections = count
                x0, y0, x1, y1 = x2, y2, x3, y3

        free_points = []

        for x, y in bresenham(x0, y0, x1, y1):
            if self.grid_map[y][x] == 0:
                free_points.append([x, y])

        if not free_points:
            return self.uniform()
        else:
            return np.array(free_points[len(free_points) // 2])


    def sample(self):
        """
        Takes sample_method input and returns the corresponding function for use in RRT class
        :return: function: The correspoding function to the sample method chosen by user
        """
        if self.sampler_type not in self.methods:
            raise ValueError(f"Unknown sampler type: {self.sampler_type}")

        return self.methods[self.sampler_type]()

class Node:
    def __init__(self, states):
        self.states = states
        self.parent = None


class RRT:
    def __init__(self, x_init, grid_map, rebuild_freq, goal, step):
        self.x_init = np.array(x_init)
        self.nodes = [Node(self.x_init)]
        self.grid_map = np.array(grid_map)
        self.map_height, self.map_width = self.grid_map.shape
        self.node_positions = [self.x_init]
        self.kd_tree = cKDTree(np.array(self.node_positions))
        self.kd_tree_needs_update = False
        self.rebuild_freq = rebuild_freq
        self.path = []
        self.smoothed_path = []
        self.goal = goal
        self.step = step
        self.sampler = None
        self.goal_reached = False
        self.node_count = 1

    def select_sampler(self, sampler_method="uniform", iterations=50, goal_bias=0.05):
        """
        Creates sampler object and sets it to self.sampler
        """
        self.sampler = Sampler(sampler_method=sampler_method,
                               iterations=iterations,
                               grid_map=self.grid_map,
                               goal=self.goal,
                               goal_bias=goal_bias,
                               height=self.map_height,
                               width=self.map_width
                               )


    def add_state(self, states, parent):
        """
        Creates node and adds node to RRT, node state also added to self.node_positions.
        Rebuilds cKDTree every rebuild frequency.
        """
        node = Node(states)
        node.parent = parent
        self.nodes.append(node)
        self.node_positions.append(states)

        # Rebuild kd-tree only after every rebuild_freq
        if len(self.nodes) % self.rebuild_freq == 0:
            self.kd_tree = cKDTree(np.array(self.node_positions))
            self.kd_tree_needs_update = False
        else:
            self.kd_tree_needs_update = True

    def find_nearest_neighbour(self, x_random):
        """
        Uses cKDTree to find the nearest node in RRT to x_random.
        Rebuilds cKDTree if neccesary
        :param x_random: Randomly sampled point on grid map
        :return: Object: Closest node to x_random
        """
        if self.kd_tree_needs_update:
            self.kd_tree = cKDTree(np.array(self.node_positions))
            self.kd_tree_needs_update = False
        _, idx = self.kd_tree.query(x_random)
        return self.nodes[idx]

    def select_control(self, x_near, x_random):
        """
        Finds the direction from node in RRT(x_near) to x_random
        :param x_near: CLosest point to x_random
        :param x_random: Randomly sampled point on grid map
        :return: array: direction towards x_random
        """
        direction = x_random - x_near.states
        norm = np.linalg.norm(direction)
        if norm == 0:
            return np.zeros_like(direction)
        return (direction / norm) * self.step

    def new_state(self, x_near, u):
        """
        Returns an adjusted value using the states of x_near to set as x_new
        :param x_near: Node in RRT closest to sampled point
        :param u: Value calculated from select_control, used to adjust x_near states
        :return: array: the states of x_new
        """
        return x_near.states + u

    def valid(self, p1, p2=None):
        """
        Checks whether a point is in an obstacle or if and edge between two points intersect with an obstacle
        :param p1: Point 1
        :param p2: Point 2
        :return: boolean: Whether the point is valid(not intersected with obstacle)
        """
        # Check nodes are in valid positions (In bounds / not in obstacles)
        if p2 is None:
            p1 = np.round(p1).astype(int)
            x_idx, y_idx = p1[0], p1[1]
            if (0 <= x_idx < self.map_width) and (0 <= y_idx < self.map_height):
                return self.grid_map[y_idx, x_idx] == 0
            return False
        # Check edges are in valid positions
        else:
            x0, y0 = np.floor(p1).astype(int)
            x1, y1 = np.floor(p2).astype(int)
            for x, y in bresenham(x0, y0, x1, y1):
                if not (0 <= x < self.map_width and 0 <= y < self.map_height):
                    return False
                if self.grid_map[y, x] == 1:
                    return False
            return True

    def grow(self, k, r):
        """
        Expands the RRT
        :param k: Number of times to sample points
        :param r: Radius to determine whether a node is close enough to the goal point
        """
        # Use algorithm from the paper
        for _ in range(k):
            x_random = self.sampler.sample()
            x_near = self.find_nearest_neighbour(x_random)
            u = self.select_control(x_near, x_random)
            x_new = self.new_state(x_near, u)

            if self.valid(x_near.states, x_new):
                self.add_state(x_new, x_near)
                self.node_count += 1

                if np.linalg.norm(x_new - self.goal) < r:
                    self.goal_reached = True
                    break

        # Ensure kd tree rebuilt at very end
        if self.kd_tree_needs_update:
            self.kd_tree = cKDTree(np.array(self.node_positions))
            self.kd_tree_needs_update = False

    def get_path(self, goal_node):
        """
        Finds path from start to goal
        :param goal_node: The node closest to the goal point
        :return: array: The path from start to goal, stored as nodes in an array
        """
        self.path = []
        current = goal_node
        while current is not None:
            self.path.append(current.states)
            current = current.parent
        self.path.reverse()
        return self.path

    def smooth_path(self):
        """
        Uses final path from get_path() and converts it into valid path with least nodes
        :return: array: Valid path with least nodes
        """
        smooth = [self.path[0]]
        i = 0
        while i < len(self.path) - 1:
            j = len(self.path) - 1
            while j > i:
                if self.valid(self.path[j], self.path[i]):
                    smooth.append(self.path[j])
                    i = j
                    break
                j -= 1
            else:
                i += 1
        self.smoothed_path = smooth
        return self.smoothed_path

    def path_cost(self):
        """
        Calculates the distance of the path from start to goal
        :return: float: Distance of final path
        """
        return sum(np.linalg.norm(self.path[i + 1] - self.path[i]) for i in range(len(self.path) - 1))

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

class Plotter:
    def __init__(self, grid_map, x_init, goal, step, rebuild_freq, k, r, sampler_method, goal_bias, iterations):
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


    def plot_grid(self):
        rrt = RRT(x_init=self.x_init, grid_map=self.grid_map, rebuild_freq=self.rebuild_freq,
                  goal=self.goal, step=self.step)
        rrt.select_sampler(sampler_method=self.sampler_method, goal_bias=self.goal_bias, iterations=self.iterations)
        rrt.grow(r=self.r, k=self.k)
        nearest_to_goal = rrt.find_nearest_neighbour(self.goal)
        path = rrt.get_path(goal_node=nearest_to_goal)

        # Draw nodes
        xs = [node.states[0] for node in rrt.nodes]
        ys = [node.states[1] for node in rrt.nodes]
        plt.scatter(xs, ys, s=5, color='gray', label='Tree Nodes')

        # Draw edges
        for node in rrt.nodes:
            if node.parent is not None:
                x1, y1 = node.states
                x2, y2 = node.parent.states
                plt.plot([x1, x2], [y1, y2], 'lightblue', linewidth=0.5)

        # Draw obstacles
        for y in range(self.grid_map.shape[0]):
            for x in range(self.grid_map.shape[1]):
                if grid_map[y, x] == 1:
                    plt.gca().add_patch(plt.Rectangle((x, y), 1, 1, color='black'))

        # PLot
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], color='red', linewidth=2, label='Path')
        plt.scatter(*self.x_init, color='green', label='Start')
        plt.scatter(*self.goal, color='blue', label='Goal')
        plt.axis('equal')
        plt.title(f"{self.sampler_method}")
        plt.grid(True)
        plt.show()

    # ========== RUNNING TEST ==========

if __name__ == "__main__":

    grid_map = load_map("grids/street-map/London_1_512.map")
    start_coords = [1, 1]
    goal_coords = np.array([378, 238])
    rebuild_freq = 500
    step = 10
    sampler_method = "goal_biased"
    goal_bias = 0.05
    iterations = 50
    k = 5000
    r = 10
    grid_map[95][95] = 0    # Make sure goal is not in an obstacle

    """
    benchmarker = Benchmark(grid_map=grid_map, x_init=start_coords, goal=goal_coords,
                            step=step, rebuild_freq=rebuild_freq, k=k, r=r)

    print(benchmarker.run_test(sampler_method="uniform", goal_bias=0.05, run_iterations=50, sample_iterations=50))

    """
    plot = Plotter(grid_map=grid_map, x_init=start_coords, goal=goal_coords,
                            step=step, rebuild_freq=rebuild_freq, k=k, r=r,
                            sampler_method=sampler_method, goal_bias=goal_bias, iterations=iterations
                   )

    plot.plot_grid()