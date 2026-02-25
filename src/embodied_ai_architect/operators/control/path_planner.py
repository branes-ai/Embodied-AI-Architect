"""A* path planner operator.

Plans collision-free paths on occupancy grids.
"""

from typing import Any
import heapq

import numpy as np

from ..base import Operator


class PathPlannerAStar(Operator):
    """A* path planner on occupancy grids.

    Plans shortest collision-free paths using A* search with
    optional diagonal movement.
    """

    def __init__(self):
        super().__init__(operator_id="path_planner_astar")
        self.grid_resolution = 0.1
        self.heuristic = "euclidean"
        self.diagonal_movement = True
        self.grid = None

    def setup(self, config: dict[str, Any], execution_target: str = "cpu") -> None:
        """Initialize path planner.

        Args:
            config: Configuration with optional keys:
                - grid_resolution: Grid cell size in meters
                - heuristic: Distance heuristic (euclidean, manhattan)
                - diagonal_movement: Allow diagonal moves
                - grid_size: Default grid size (rows, cols)
            execution_target: Only cpu supported
        """
        if execution_target != "cpu":
            print(f"[PathPlannerAStar] Warning: Only CPU supported")

        self._execution_target = "cpu"
        self._config = config

        self.grid_resolution = config.get("grid_resolution", 0.1)
        self.heuristic = config.get("heuristic", "euclidean")
        self.diagonal_movement = config.get("diagonal_movement", True)

        # Initialize default grid if specified
        grid_size = config.get("grid_size", (100, 100))
        self.grid = np.zeros(grid_size, dtype=np.uint8)

        self._is_setup = True
        print(f"[PathPlannerAStar] Ready (resolution={self.grid_resolution}m)")

    def process(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Plan path from start to goal.

        Args:
            inputs: Dictionary with:
                - 'start': Start position (x, y) in meters or grid coords
                - 'goal': Goal position (x, y) or detection list (uses first detection bbox center)
                - 'detections': Alternative way to specify goal via detection list
                - 'grid': Optional occupancy grid (0=free, >0=obstacle)
                - 'use_meters': If True, convert start/goal from meters

        Returns:
            Dictionary with:
                - 'path': List of waypoints [[x, y], ...]
                - 'success': Whether path was found
                - 'path_length': Number of waypoints
        """
        start = inputs.get("start", (0, 0))
        goal = inputs.get("goal")
        grid = inputs.get("grid", self.grid)
        use_meters = inputs.get("use_meters", False)

        # Handle goal from detections (pick_and_place architecture)
        if goal is None:
            detections = inputs.get("detections", [])
            if detections and len(detections) > 0:
                # Use center of first detection as goal
                det = detections[0]
                bbox = det.get("bbox", [0, 0, 0, 0])
                goal = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            else:
                goal = (50, 50)  # Default to grid center
        elif isinstance(goal, list) and len(goal) > 0 and isinstance(goal[0], dict):
            # Goal is a list of detections
            det = goal[0]
            bbox = det.get("bbox", [0, 0, 0, 0])
            goal = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        elif not goal or (hasattr(goal, "__len__") and len(goal) < 2):
            goal = (50, 50)  # Default

        if use_meters:
            start = self._meters_to_grid(start)
            goal = self._meters_to_grid(goal)

        # Ensure integer grid coordinates
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))

        path, success = self._astar(grid, start, goal)

        if success and use_meters:
            path = [self._grid_to_meters(p) for p in path]

        return {
            "path": path if path else [],
            "success": success,
            "path_length": len(path) if path else 0,
        }

    def _meters_to_grid(self, pos: tuple) -> tuple:
        """Convert meters to grid coordinates."""
        return (
            int(pos[0] / self.grid_resolution),
            int(pos[1] / self.grid_resolution),
        )

    def _grid_to_meters(self, pos: tuple) -> list:
        """Convert grid coordinates to meters."""
        return [
            pos[0] * self.grid_resolution,
            pos[1] * self.grid_resolution,
        ]

    def _astar(
        self,
        grid: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> tuple[list | None, bool]:
        """A* path planning algorithm."""
        rows, cols = grid.shape

        # Validate start and goal
        if not (0 <= start[0] < rows and 0 <= start[1] < cols):
            return None, False
        if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
            return None, False
        if grid[start[0], start[1]] > 0:
            return None, False
        if grid[goal[0], goal[1]] > 0:
            return None, False

        # Priority queue: (f_score, counter, position)
        counter = 0
        open_set = [(0, counter, start)]
        came_from: dict[tuple, tuple] = {}

        g_score: dict[tuple, float] = {start: 0}
        f_score: dict[tuple, float] = {start: self._heuristic(start, goal)}

        closed_set: set[tuple] = set()

        # Movement directions
        if self.diagonal_movement:
            directions = [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]
            costs = [1.414, 1, 1.414, 1, 1, 1.414, 1, 1.414]
        else:
            directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
            costs = [1, 1, 1, 1]

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path)), True

            if current in closed_set:
                continue
            closed_set.add(current)

            for (dr, dc), cost in zip(directions, costs):
                neighbor = (current[0] + dr, current[1] + dc)

                # Check bounds
                if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                    continue

                # Check obstacle
                if grid[neighbor[0], neighbor[1]] > 0:
                    continue

                # Check if already processed
                if neighbor in closed_set:
                    continue

                tentative_g = g_score[current] + cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor))

        return None, False

    def _heuristic(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        """Calculate heuristic distance."""
        if self.heuristic == "manhattan":
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        else:  # euclidean
            return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def update_grid(self, grid: np.ndarray) -> None:
        """Update the internal occupancy grid."""
        self.grid = grid

    def teardown(self) -> None:
        """Clean up."""
        self.grid = None
        self._is_setup = False
