import numpy as np
from scipy.spatial import cKDTree
from sampler import Sampler, bresenham


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
        self.best_cost = None

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

    def informed_grow(self, k, r):
        """
        Grows the RRT through informed sampling
        Finds first path using grow() then perform informed sampling to find final path
        :param k: Number of times to sample
        :param r: Radius of goal region
        """
        while not self.goal_reached:
            self.grow(k, r)

        print("Initial path found")

        # Update path and best cost
        goal = self.find_nearest_neighbour(self.goal)
        start = Node(self.x_init)
        self.get_path(goal)
        self.best_cost = self.path_cost()

        for _ in range(k):
            # Use informed sampling
            x_random = self.sampler.informed(start=start, goal=goal, best_cost=self.best_cost)

            # Get new node
            x_near = self.find_nearest_neighbour(x_random)
            u = self.select_control(x_near, x_random)
            x_new = self.new_state(x_near, u)

            # Check if new node is valid
            if self.valid(x_near.states, x_new):
                self.add_state(x_new, x_near)
                self.node_count += 1

                if np.linalg.norm(x_new - self.goal) < r:
                    self.get_path(self.find_nearest_neighbour(self.goal))
                    new_cost = self.path_cost()
                    if new_cost < self.best_cost:
                        self.best_cost = new_cost

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

