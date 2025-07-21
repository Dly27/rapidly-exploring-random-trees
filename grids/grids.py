# maps: https://movingai.com/benchmarks/grids.html

import numpy as np

def narrow_passage_map(size=100):
    grid = np.zeros((size, size), dtype=np.uint8)
    grid[:, 45:55] = 1
    grid[59:60, 45:55] = 0  # narrow opening
    return grid

def u_maze_map(size=100):
    grid = np.zeros((size, size), dtype=np.uint8)
    grid[20:80, 40] = 1
    grid[20:80, 60] = 1
    grid[20, 40:60] = 1
    grid[80, 40:60] = 1
    return grid

def random_cluttered_map(size=100, n_rects=15, min_size=5, max_size=15):
    grid = np.zeros((size, size), dtype=np.uint8)
    for _ in range(n_rects):
        h = np.random.randint(min_size, max_size)
        w = np.random.randint(min_size, max_size)
        x = np.random.randint(0, size - h)
        y = np.random.randint(0, size - w)
        grid[x:x+h, y:y+w] = 1
    return grid

def load_map(file):
    """
    Loads .map file and converts into desired grid map array format
    :param file: The file loaded from \street-map
    :return: array: The grid map used for RRT
    """
    with open(file, "r") as f:
        lines = f.readlines()

    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])
    map_lines = lines[4:4 + height]

    grid = np.zeros((height, width), dtype=np.uint8)

    for i, line in enumerate(map_lines):
        for j, char in enumerate(line.strip()):
            if char in ['@', 'T', 'W', '%']:
                grid[i, j] = 1
            else:
                grid[i, j] = 0

    return np.array(grid)