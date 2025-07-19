# Rapidly-exploring random trees path planning

This project implements the rapidly-exploring random tree (RRT) algorithm for path planning on a 2D space with obstacles, based on:

- [LaValle, 1998 - Rapidly-exploring random trees: A new tool for path planning](https://msl.cs.illinois.edu/~lavalle/papers/Lav98c.pdf)
- [Karaman & Frazzoli, 2011 - Sampling-based algorithms for optimal motion planning](https://arxiv.org/pdf/1105.1186)

## Requirements

- Python 3.11
- `numpy`, `scipy`, `matplotlib`

## Features

Main algorithm:

- Implements RRT with goal biasing
- Edge validation through Bresenham's line algorithm to check whether are not in obstacles

Environment:

- 2D grid map, `grid_map` with obstacles determined by binary encoding
- Obsactles set by defining a value in `grid_map` as 1

Path planning:

- Fast nearest neighbour search through SciPy's `cKDtree`
- Shortest path extraction from start to end using parent pointers
- Path smoothing through shortcutting while still collision safe
- Path cost compuation

Visualisation:

- Plot displaying: nodes, edges, final path, obstacles, start and end points

## Usage

```bash
pip install numpy matplotlib scipy
python rrt.py
```

This will run the RRT algorithm on a 100x100 grid with a predetermined obstacle set, which should include: all nodes and edges, a red path indicating the final path from start to end, obstacles indicated as black structures, a start(green) and end(blue) node,

  
