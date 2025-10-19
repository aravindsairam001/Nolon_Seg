"""
Trajectory planning pipeline that integrates floor detection, obstacle detection, and path planning.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
import time

from .obstacle_detector import ObstacleDetector
from .path_planner import TrajectoryPlanner
from .robot_config import RobotConfig, create_robot_config

class TrajectoryPlanningPipeline:
    """Complete trajectory planning pipeline for robot navigation."""
    
    def __init__(self, 
                 robot_config: Optional[RobotConfig] = None,
                 enable_dynamic_obstacles: bool = True,
                 enable_path_smoothing: bool = True):
        """
        Initialize trajectory planning pipeline.
        
        Args:
            robot_config: Robot configuration parameters
            enable_dynamic_obstacles: Enable detection of moving obstacles
            enable_path_smoothing: Enable path smoothing
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize robot configuration
        if robot_config is None:
            robot_config = create_robot_config()
        self.robot_config = robot_config
        
        # Initialize components
        self.obstacle_detector = ObstacleDetector(
            min_obstacle_area=100,
            max_obstacle_area=50000
        )
        
        self.trajectory_planner = TrajectoryPlanner(
            robot_radius=robot_config.get('robot_radius'),
            safety_margin=robot_config.get('safety_margin'),
            grid_resolution=robot_config.get('grid_resolution'),
            lookahead_distance=robot_config.get('planning_horizon')
        )
        
        # Pipeline settings
        self.enable_dynamic_obstacles = enable_dynamic_obstacles
        self.enable_path_smoothing = enable_path_smoothing
        
        # State tracking
        self.previous_frame = None
        self.previous_trajectory = None
        self.trajectory_history = []
        
    def process_frame(self,
                     frame: np.ndarray,
                     floor_mask: np.ndarray,
                     robot_position: Optional[Tuple[int, int]] = None,
                     goal_direction: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Process a single frame to generate trajectory.
        
        Args:
            frame: Input video frame (BGR format)
            floor_mask: Binary mask of drivable floor area
            robot_position: Current robot position (row, col), or None for center
            goal_direction: Preferred movement direction, or None for forward
            
        Returns:
            Dictionary containing trajectory and analysis results
        """
        start_time = time.time()
        
        # Default robot position to center bottom of frame
        if robot_position is None:
            height, width = frame.shape[:2]
            robot_position = (int(height * 0.8), width // 2)
        
        # Step 1: Detect obstacles
        obstacle_mask = self.obstacle_detector.detect_obstacles(frame, floor_mask)
        
        # Step 2: Detect dynamic obstacles if enabled
        dynamic_obstacle_mask = np.zeros_like(obstacle_mask)
        if self.enable_dynamic_obstacles and self.previous_frame is not None:
            dynamic_obstacle_mask = self.obstacle_detector.detect_dynamic_obstacles(
                frame, self.previous_frame, floor_mask
            )
            # Combine static and dynamic obstacles
            obstacle_mask = obstacle_mask | dynamic_obstacle_mask
        
        # Step 3: Plan trajectory
        trajectory = self.trajectory_planner.plan_trajectory(
            floor_mask=floor_mask,
            obstacle_mask=obstacle_mask,
            robot_position=robot_position,
            goal_direction=goal_direction
        )
        
        # Step 4: Analyze trajectory safety
        safety_analysis = self._analyze_trajectory_safety(
            trajectory, obstacle_mask, frame.shape[:2]
        )
        
        # Step 5: Convert to world coordinates
        world_trajectory = None
        if trajectory is not None:
            world_trajectory = self.trajectory_planner.convert_path_to_world_coordinates(
                trajectory, frame.shape[:2]
            )
        
        # Step 6: Calculate trajectory metrics
        metrics = self._calculate_trajectory_metrics(
            trajectory, obstacle_mask, frame.shape[:2]
        )
        
        # Update state
        self.previous_frame = frame.copy()
        self.previous_trajectory = trajectory
        if trajectory is not None:
            self.trajectory_history.append(trajectory)
            
        # Keep history bounded
        max_history = 10
        if len(self.trajectory_history) > max_history:
            self.trajectory_history = self.trajectory_history[-max_history:]
        
        processing_time = time.time() - start_time
        
        # Return comprehensive results
        results = {
            'trajectory': trajectory,
            'world_trajectory': world_trajectory,
            'obstacle_mask': obstacle_mask,
            'dynamic_obstacle_mask': dynamic_obstacle_mask,
            'floor_mask': floor_mask,
            'robot_position': robot_position,
            'safety_analysis': safety_analysis,
            'metrics': metrics,
            'processing_time': processing_time,
            'frame_info': {
                'shape': frame.shape,
                'timestamp': time.time()
            }
        }
        
        return results
    
    def _analyze_trajectory_safety(self,
                                  trajectory: Optional[List[Tuple[int, int]]],
                                  obstacle_mask: np.ndarray,
                                  frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Analyze trajectory safety."""
        analysis = {
            'is_safe': False,
            'collision_risk': 'unknown',
            'clear_distance': 0.0,
            'obstacles_nearby': False,
            'emergency_stop_required': False,
            'confidence': 0.0
        }
        
        if trajectory is None or len(trajectory) == 0:
            analysis['collision_risk'] = 'no_path'
            return analysis
        
        # Calculate distance map from obstacles
        distance_map = self.obstacle_detector.create_obstacle_distance_map(obstacle_mask)
        
        # Check trajectory safety
        min_distance = float('inf')
        collision_points = 0
        
        robot_radius_pixels = int(self.robot_config.get_effective_radius() / 
                                self.robot_config.get_planning_resolution())
        
        for point in trajectory:
            row, col = point
            if 0 <= row < distance_map.shape[0] and 0 <= col < distance_map.shape[1]:
                distance = distance_map[row, col]
                min_distance = min(min_distance, distance)
                
                if distance < robot_radius_pixels:
                    collision_points += 1
        
        # Determine safety status
        analysis['clear_distance'] = min_distance * self.robot_config.get_planning_resolution()
        analysis['is_safe'] = collision_points == 0
        analysis['obstacles_nearby'] = min_distance < robot_radius_pixels * 2
        analysis['emergency_stop_required'] = self.robot_config.requires_emergency_stop(
            analysis['clear_distance']
        )
        
        # Risk assessment
        if collision_points > 0:
            analysis['collision_risk'] = 'high'
        elif analysis['obstacles_nearby']:
            analysis['collision_risk'] = 'medium'
        else:
            analysis['collision_risk'] = 'low'
        
        # Confidence based on trajectory length and clearance
        if trajectory is not None:
            trajectory_length = len(trajectory)
            max_expected_length = int(self.robot_config.get('planning_horizon') / 
                                    self.robot_config.get_planning_resolution())
            length_factor = min(1.0, trajectory_length / max_expected_length)
            distance_factor = min(1.0, analysis['clear_distance'] / 
                                self.robot_config.get('min_obstacle_distance'))
            analysis['confidence'] = (length_factor + distance_factor) / 2
        
        return analysis
    
    def _calculate_trajectory_metrics(self,
                                    trajectory: Optional[List[Tuple[int, int]]],
                                    obstacle_mask: np.ndarray,
                                    frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Calculate trajectory quality metrics."""
        metrics = {
            'length': 0.0,
            'straightness': 0.0,
            'smoothness': 0.0,
            'obstacle_clearance': 0.0,
            'execution_feasibility': 0.0,
            'overall_quality': 0.0
        }
        
        if trajectory is None or len(trajectory) < 2:
            return metrics
        
        # Calculate trajectory length
        total_length = 0.0
        for i in range(1, len(trajectory)):
            p1 = trajectory[i-1]
            p2 = trajectory[i]
            distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            total_length += distance
        
        metrics['length'] = total_length * self.robot_config.get_planning_resolution()
        
        # Calculate straightness (ratio of direct distance to path length)
        if len(trajectory) >= 2:
            start = trajectory[0]
            end = trajectory[-1]
            direct_distance = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            direct_distance *= self.robot_config.get_planning_resolution()
            
            if total_length > 0:
                metrics['straightness'] = direct_distance / metrics['length']
        
        # Calculate smoothness (based on direction changes)
        direction_changes = 0
        if len(trajectory) >= 3:
            for i in range(1, len(trajectory) - 1):
                p1, p2, p3 = trajectory[i-1], trajectory[i], trajectory[i+1]
                
                # Calculate vectors
                v1 = (p2[0] - p1[0], p2[1] - p1[1])
                v2 = (p3[0] - p2[0], p3[1] - p2[1])
                
                # Calculate angle between vectors
                dot_product = v1[0]*v2[0] + v1[1]*v2[1]
                magnitude1 = np.sqrt(v1[0]**2 + v1[1]**2)
                magnitude2 = np.sqrt(v2[0]**2 + v2[1]**2)
                
                if magnitude1 > 0 and magnitude2 > 0:
                    cos_angle = dot_product / (magnitude1 * magnitude2)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)
                    direction_changes += angle
            
            # Normalize smoothness (lower direction changes = higher smoothness)
            avg_direction_change = direction_changes / (len(trajectory) - 2)
            metrics['smoothness'] = max(0, 1 - avg_direction_change / np.pi)
        
        # Calculate obstacle clearance
        distance_map = self.obstacle_detector.create_obstacle_distance_map(obstacle_mask)
        clearances = []
        
        for point in trajectory:
            row, col = point
            if 0 <= row < distance_map.shape[0] and 0 <= col < distance_map.shape[1]:
                clearance = distance_map[row, col] * self.robot_config.get_planning_resolution()
                clearances.append(clearance)
        
        if clearances:
            metrics['obstacle_clearance'] = np.mean(clearances)
        
        # Calculate execution feasibility based on robot constraints
        max_velocity = self.robot_config.get('max_linear_velocity')
        feasible_points = 0
        
        for i in range(1, len(trajectory)):
            # Check if robot can execute this segment
            # This is a simplified check - could be enhanced with dynamics
            feasible_points += 1
        
        metrics['execution_feasibility'] = feasible_points / len(trajectory) if trajectory else 0
        
        # Calculate overall quality score
        weights = {
            'straightness': 0.2,
            'smoothness': 0.3,
            'obstacle_clearance': 0.3,
            'execution_feasibility': 0.2
        }
        
        overall_quality = 0.0
        for metric, weight in weights.items():
            overall_quality += metrics[metric] * weight
        
        metrics['overall_quality'] = overall_quality
        
        return metrics
    
    def get_robot_config(self) -> RobotConfig:
        """Get current robot configuration."""
        return self.robot_config
    
    def update_robot_config(self, new_config: RobotConfig):
        """Update robot configuration."""
        self.robot_config = new_config
        
        # Update dependent components
        self.trajectory_planner = TrajectoryPlanner(
            robot_radius=new_config.get('robot_radius'),
            safety_margin=new_config.get('safety_margin'),
            grid_resolution=new_config.get('grid_resolution'),
            lookahead_distance=new_config.get('planning_horizon')
        )
        
        self.logger.info("Updated robot configuration and trajectory planner")
    
    def get_trajectory_history(self) -> List[List[Tuple[int, int]]]:
        """Get recent trajectory history."""
        return self.trajectory_history.copy()
    
    def clear_history(self):
        """Clear trajectory history."""
        self.trajectory_history.clear()
        self.previous_frame = None
        self.previous_trajectory = None
        self.logger.info("Cleared trajectory history")