from collections import deque

import numpy as np


def get_waypoints(maze_arena):
    waypoints = []
    start = maze_arena.spawn_positions[:2]
    start = maze_arena.world_to_grid_positions(start)
    start = np.asarray(start, dtype=np.int32)[0][:2]
    goal = maze_arena.target_positions[:2]
    goal = maze_arena.world_to_grid_positions(goal)
    goal = np.asarray(goal, dtype=np.int32)[0][:2]

    maze = maze_arena.maze.entity_layer

    queue = deque()
    queue.append((start[0], start[1]))
    visited = {(start[0], start[1])}
    previous = {(start[0], start[1]): None}
    while True:
        top = queue.popleft()
        if top[0] == goal[0] and top[1] == goal[1]:
            break

        for di, dj in zip([1, -1, 0, 0], [0, 0, 1, -1]):
            new_i = top[0] + di
            new_j = top[1] + dj
            if new_i >= 0 and new_i < maze.shape[
                    0] and new_j >= 0 and new_j < maze.shape[1] and maze[
                        new_i, new_j] != '*' and not (new_i, new_j) in visited:
                queue.append((new_i, new_j))
                visited.add((new_i, new_j))
                previous[(new_i, new_j)] = (top[0], top[1])

    last = top
    while last is not None:
        waypoints.append(last)
        last = previous[last]

    waypoints = list(reversed(waypoints))
    return maze_arena.grid_to_world_positions(waypoints)
