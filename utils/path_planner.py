"""
A* path planning algorithm for robot trajectory generation.
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional, Set
import logging
import math

class AStarPlanner:
    """A* path planning algorithm for 2D grid-based navigation."""
    
    def __init__(self, robot_radius: float = 0.3, safety_margin: float = 0.2):
        """
        Initialize A* planner.
        
        Args:
            robot_radius: Robot radius in meters
            safety_margin: Additional safety margin in meters
        """
        self.logger = logging.getLogger(__name__)
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin
        self.total_radius = robot_radius + safety_margin
        
        # 8-connected movement (including diagonals)
        self.movements = [
            (-1, -1, math.sqrt(2)), (-1, 0, 1), (-1, 1, math.sqrt(2)),
            (0, -1, 1),                        (0, 1, 1),
            (1, -1, math.sqrt(2)),  (1, 0, 1),  (1, 1, math.sqrt(2))
        ]
    
    def plan_path(self, 
                  occupancy_grid: np.ndarray,
                  start: Tuple[int, int],
                  goal: Tuple[int, int],
                  grid_resolution: float = 0.05) -> Optional[List[Tuple[int, int]]]:
        """
        Plan a path using A* algorithm.
        
        Args:
            occupancy_grid: 2D grid where 0=free, 1=obstacle
            start: Start position (row, col)
            goal: Goal position (row, col)
            grid_resolution: Resolution of grid in meters/pixel
            
        Returns:
            List of waypoints from start to goal, or None if no path found
        """
        if not self._is_valid_position(occupancy_grid, start):
            self.logger.error(f"Invalid start position: {start}")
            return None
            
        if not self._is_valid_position(occupancy_grid, goal):
            self.logger.error(f"Invalid goal position: {goal}")
            return None
        
        # Calculate inflation radius in grid cells
        inflation_cells = int(self.total_radius / grid_resolution)
        
        # Inflate obstacles
        inflated_grid = self._inflate_obstacles(occupancy_grid, inflation_cells)
        
        # Run A* algorithm
        path = self._astar_search(inflated_grid, start, goal)
        
        if path is None:
            self.logger.warning(f"No path found from {start} to {goal}")
            return None
        
        # Smooth the path
        smoothed_path = self._smooth_path(path, inflated_grid)
        
        return smoothed_path
    
    def _astar_search(self, 
                     grid: np.ndarray, 
                     start: Tuple[int, int], 
                     goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Core A* search algorithm."""
        rows, cols = grid.shape
        
        # Priority queue: (f_score, g_score, position)
        open_set = [(0, 0, start)]
        heapq.heapify(open_set)
        
        # Dictionaries to track costs and paths
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        came_from = {}
        closed_set: Set[Tuple[int, int]] = set()
        
        while open_set:
            # Get position with lowest f_score
            current_f, current_g, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
                
            closed_set.add(current)
            
            # Check if we reached the goal
            if current == goal:
                return self._reconstruct_path(came_from, current)
            
            # Explore neighbors
            for dx, dy, cost in self.movements:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check bounds
                if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                    continue
                
                # Check if neighbor is blocked
                if grid[neighbor[0], neighbor[1]] > 0:
                    continue
                
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g_score
                tentative_g = g_score[current] + cost
                
                # If this path to neighbor is better than previous, record it
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    
                    # Add to open set if not already there
                    heapq.heappush(open_set, (f_score[neighbor], tentative_g, neighbor))
        
        return None  # No path found
    
    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Euclidean distance heuristic."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _is_valid_position(self, grid: np.ndarray, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds and not blocked."""
        rows, cols = grid.shape
        row, col = pos
        
        if not (0 <= row < rows and 0 <= col < cols):
            return False
        
        return grid[row, col] == 0
    
    def _inflate_obstacles(self, grid: np.ndarray, inflation_radius: int) -> np.ndarray:
        """Inflate obstacles by robot radius for safe planning."""
        if inflation_radius <= 0:
            return grid.copy()
        
        # Create structuring element for inflation
        kernel_size = 2 * inflation_radius + 1
        y, x = np.ogrid[:kernel_size, :kernel_size]
        mask = (x - inflation_radius)**2 + (y - inflation_radius)**2 <= inflation_radius**2
        kernel = mask.astype(np.uint8)
        
        # Dilate obstacles
        import cv2
        inflated = cv2.dilate(grid.astype(np.uint8), kernel, iterations=1)
        
        return inflated
    
    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from goal to start."""
        path = [current]
        
        while current in came_from:
            current = came_from[current]
            path.append(current)
        
        path.reverse()
        return path
    
    def _smooth_path(self, path: List[Tuple[int, int]], grid: np.ndarray) -> List[Tuple[int, int]]:
        """Smooth path by removing unnecessary waypoints."""
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            # Find the farthest point we can reach directly
            farthest_idx = current_idx + 1
            
            for i in range(current_idx + 2, len(path)):
                if self._is_line_clear(grid, path[current_idx], path[i]):
                    farthest_idx = i
                else:
                    break
            
            # Add the farthest reachable point
            if farthest_idx != current_idx:
                smoothed.append(path[farthest_idx])
                current_idx = farthest_idx
            else:
                # If we can't skip any points, move to next
                current_idx += 1
                if current_idx < len(path):
                    smoothed.append(path[current_idx])
        
        return smoothed
    
    def _is_line_clear(self, grid: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """Check if line between two points is clear of obstacles."""
        # Bresenham's line algorithm
        x0, y0 = start
        x1, y1 = end
        
        points = self._bresenham_line(x0, y0, x1, y1)
        
        for x, y in points:
            if not (0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]):
                return False
            if grid[x, y] > 0:
                return False
        
        return True
    
    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Bresenham's line algorithm to get points along a line."""
        points = []
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            points.append((x, y))
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            
            if e2 > -dy:
                err -= dy
                x += sx
            
            if e2 < dx:
                err += dx
                y += sy
        
        return points

class TrajectoryPlanner:
    """High-level trajectory planner that uses A* for path planning."""
    
    def __init__(self, 
                 robot_radius: float = 0.3,
                 safety_margin: float = 0.2,
                 grid_resolution: float = 0.05,
                 lookahead_distance: float = 3.0):
        """
        Initialize trajectory planner.
        
        Args:
            robot_radius: Robot radius in meters
            safety_margin: Additional safety margin in meters
            grid_resolution: Grid resolution in meters/pixel
            lookahead_distance: How far ahead to plan in meters
        """
        self.logger = logging.getLogger(__name__)
        self.astar = AStarPlanner(robot_radius, safety_margin)
        self.grid_resolution = grid_resolution
        self.lookahead_distance = lookahead_distance
    
    def plan_trajectory(self,
                       floor_mask: np.ndarray,
                       obstacle_mask: np.ndarray,
                       robot_position: Tuple[int, int],
                       goal_direction: Optional[Tuple[float, float]] = None) -> Optional[List[Tuple[int, int]]]:
        """
        Plan trajectory from robot position toward goal.
        
        Args:
            floor_mask: Binary mask of drivable floor area
            obstacle_mask: Binary mask of obstacles
            robot_position: Current robot position (row, col)
            goal_direction: Preferred direction as (dx, dy), or None for forward
            
        Returns:
            List of trajectory waypoints or None if no valid path
        """
        # Create occupancy grid
        occupancy_grid = self._create_occupancy_grid(floor_mask, obstacle_mask)
        
        # Determine goal position
        goal_position = self._get_goal_position(
            robot_position, 
            goal_direction, 
            occupancy_grid
        )
        
        if goal_position is None:
            self.logger.warning("No valid goal position found")
            return None
        
        # Plan path using A*
        path = self.astar.plan_path(
            occupancy_grid,
            robot_position,
            goal_position,
            self.grid_resolution
        )
        
        return path
    
    def _create_occupancy_grid(self, floor_mask: np.ndarray, obstacle_mask: np.ndarray) -> np.ndarray:
        """Create occupancy grid from floor and obstacle masks."""
        # Start with all cells as obstacles
        occupancy_grid = np.ones_like(floor_mask, dtype=np.uint8)
        
        # Mark floor areas as free space
        occupancy_grid[floor_mask] = 0
        
        # Mark obstacles as blocked
        occupancy_grid[obstacle_mask] = 1
        
        return occupancy_grid
    
    def _get_goal_position(self,
                          robot_position: Tuple[int, int],
                          goal_direction: Optional[Tuple[float, float]],
                          occupancy_grid: np.ndarray) -> Optional[Tuple[int, int]]:
        """Determine goal position based on lookahead distance and direction."""
        rows, cols = occupancy_grid.shape
        robot_row, robot_col = robot_position
        
        # Default direction is forward (down in image coordinates)
        if goal_direction is None:
            goal_direction = (1.0, 0.0)  # Move down in image
        
        dx, dy = goal_direction
        
        # Normalize direction
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            dx /= length
            dy /= length
        
        # Calculate goal position based on lookahead distance
        lookahead_pixels = int(self.lookahead_distance / self.grid_resolution)
        
        goal_row = robot_row + int(dx * lookahead_pixels)
        goal_col = robot_col + int(dy * lookahead_pixels)
        
        # Ensure goal is within bounds
        goal_row = max(0, min(rows - 1, goal_row))
        goal_col = max(0, min(cols - 1, goal_col))
        
        # Find nearest free space if goal is in obstacle
        goal_position = self._find_nearest_free_space(
            occupancy_grid, 
            (goal_row, goal_col)
        )
        
        return goal_position
    
    def _find_nearest_free_space(self, 
                                occupancy_grid: np.ndarray, 
                                target: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find nearest free space to target position."""
        rows, cols = occupancy_grid.shape
        target_row, target_col = target
        
        # If target is already free, return it
        if (0 <= target_row < rows and 0 <= target_col < cols and 
            occupancy_grid[target_row, target_col] == 0):
            return target
        
        # Search in expanding circles
        max_search_radius = min(rows, cols) // 4
        
        for radius in range(1, max_search_radius):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if abs(dr) != radius and abs(dc) != radius:
                        continue  # Only check perimeter
                    
                    test_row = target_row + dr
                    test_col = target_col + dc
                    
                    if (0 <= test_row < rows and 0 <= test_col < cols and
                        occupancy_grid[test_row, test_col] == 0):
                        return (test_row, test_col)
        
        return None
    
    def convert_path_to_world_coordinates(self, 
                                        path: List[Tuple[int, int]], 
                                        image_shape: Tuple[int, int],
                                        real_world_width: float = 6.0) -> List[Tuple[float, float]]:
        """
        Convert pixel path to real-world coordinates.
        
        Args:
            path: Path in pixel coordinates
            image_shape: Shape of the image (height, width)
            real_world_width: Real world width represented by image width
            
        Returns:
            Path in real-world coordinates (x, y in meters)
        """
        height, width = image_shape
        scale = real_world_width / width
        
        world_path = []
        for row, col in path:
            # Convert to world coordinates (origin at bottom-left)
            world_x = col * scale
            world_y = (height - row) * scale
            world_path.append((world_x, world_y))
        
        return world_path