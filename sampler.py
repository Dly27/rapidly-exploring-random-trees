import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from scipy.stats.qmc import Halton


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
        """
        Prevents sampled points to round outside of grid bounds
        :param point: Sampled point
        :return: array: Rounded point
        """
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
        """
        Finds distance from a point to the closest obstacle
        :param point: Sampled point
        :return: float: Distance from point to closest obstacle
        """
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
        x = point[0] * self.width
        y = point[1] * self.height

        return np.array([x, y])

    def informed(self, start, goal, best_cost):
        """
        Calculates properties of an ellipsoid based on the start and end point
        Samples a point from a unit circle
        Transforms the unit circle / sampled point into an ellipsoid as the final sample
        :param start: Start node
        :param goal: Goal node
        :param best_cost: Lowest path cost found so far in RRT
        :return: array: Sample point
        """
        # Ellipsoid properties
        start, goal = start.states, goal.states
        c_min = np.linalg.norm(goal - start)
        centre = (start + goal) / 2

        # Direction vector from start to goal
        a1 = (goal - start) / c_min

        # Create a basis for the ellipsoid using a rotation matrix C
        # First axis is a1, other axes are perpendicular directions

        # Construct orthonormal basis
        dim = start.shape[0]
        U, _, Vt = np.linalg.svd(np.eye(dim))  # identity matrix for base axes

        # Rotation matrix C to align the x-axis to a1
        C = np.zeros((dim, dim))
        C[:, 0] = a1
        C[:, 1:] = U[:, 1:]

        # Radii of the ellipsoid axes
        r1 = best_cost / 2
        if r1 ** 2 - (c_min / 2) ** 2 < 0:
            # Safety fallback
            r2 = 0
        else:
            r2 = np.sqrt(r1 ** 2 - (c_min / 2) ** 2)
        radii = np.array([r1 * 0.5] + [r2 * 0.5] * (dim - 1))

        def sample_unit_n_ball(d):
            x = np.random.normal(0, 1, d)
            x /= np.linalg.norm(x)
            radius = np.random.rand() ** (1.0 / d)
            return radius * x

        x_ball = sample_unit_n_ball(dim)

        # Scale by radii and rotate
        sample = centre + C @ (radii * x_ball)
        return sample

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
