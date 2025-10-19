"""
Robot configuration and parameters for trajectory planning.
"""

import numpy as np
from typing import Dict, Any, Tuple
import logging

class RobotConfig:
    """Configuration parameters for robot trajectory planning."""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Initialize robot configuration.
        
        Args:
            config_dict: Dictionary of configuration parameters
        """
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        default_config = {
            # Physical parameters
            'robot_width': 3.0,          # Robot width in meters (default: 3m)
            'robot_length': 3.0,         # Robot length in meters
            'robot_radius': 1.5,         # Effective robot radius for planning
            'wheel_base': 2.0,           # Distance between front and rear axles
            
            # Safety parameters
            'safety_margin': 0.5,        # Additional safety margin around robot
            'min_obstacle_distance': 1.0, # Minimum distance to maintain from obstacles
            'emergency_stop_distance': 0.8, # Distance for emergency stop
            
            # Motion parameters
            'max_linear_velocity': 1.0,   # Maximum linear velocity (m/s)
            'max_angular_velocity': 1.0,  # Maximum angular velocity (rad/s)
            'max_linear_acceleration': 0.5, # Maximum linear acceleration (m/s²)
            'max_angular_acceleration': 1.0, # Maximum angular acceleration (rad/s²)
            
            # Planning parameters
            'planning_horizon': 3.0,      # Planning horizon in meters
            'grid_resolution': 0.05,      # Grid resolution for planning (m/pixel)
            'path_smoothing': True,       # Enable path smoothing
            'dynamic_replanning': True,   # Enable dynamic replanning
            
            # Trajectory parameters
            'trajectory_update_rate': 10.0, # Hz - how often to update trajectory
            'waypoint_tolerance': 0.1,     # Distance tolerance for waypoint completion
            'goal_tolerance': 0.2,         # Distance tolerance for goal completion
            'max_trajectory_length': 50,   # Maximum number of trajectory points
            
            # Camera and sensing parameters
            'camera_height': 1.2,         # Camera height from ground (meters)
            'camera_tilt_angle': 0.0,     # Camera tilt angle (radians)
            'field_of_view': 60.0,        # Camera field of view (degrees)
            'sensing_range': 5.0,         # Maximum sensing range (meters)
            
            # Environment parameters
            'floor_height_threshold': 0.1, # Maximum height variation for floor
            'obstacle_height_threshold': 0.2, # Minimum height for obstacles
            'ground_clearance': 0.05,     # Robot ground clearance
        }
        
        # Update with provided config
        self.config = default_config.copy()
        if config_dict:
            self.config.update(config_dict)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Check for negative values that should be positive
        positive_params = [
            'robot_width', 'robot_length', 'robot_radius', 'wheel_base',
            'safety_margin', 'min_obstacle_distance', 'emergency_stop_distance',
            'max_linear_velocity', 'max_angular_velocity', 
            'max_linear_acceleration', 'max_angular_acceleration',
            'planning_horizon', 'grid_resolution', 'trajectory_update_rate',
            'waypoint_tolerance', 'goal_tolerance', 'camera_height',
            'field_of_view', 'sensing_range', 'ground_clearance'
        ]
        
        for param in positive_params:
            if self.config[param] <= 0:
                self.logger.warning(f"Parameter {param} should be positive, got {self.config[param]}")
        
        # Check robot radius is reasonable compared to width/length
        if self.config['robot_radius'] < max(self.config['robot_width'], self.config['robot_length']) / 2:
            self.logger.warning("Robot radius might be too small compared to robot dimensions")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration parameter."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration parameter."""
        self.config[key] = value
        self.logger.info(f"Updated config: {key} = {value}")
    
    def get_robot_footprint(self) -> np.ndarray:
        """Get robot footprint as polygon vertices."""
        width = self.config['robot_width'] / 2
        length = self.config['robot_length'] / 2
        
        # Rectangle footprint centered at origin
        footprint = np.array([
            [-length, -width],
            [length, -width],
            [length, width],
            [-length, width],
            [-length, -width]  # Close the polygon
        ])
        
        return footprint
    
    def get_safety_footprint(self) -> np.ndarray:
        """Get robot footprint including safety margin."""
        width = (self.config['robot_width'] + 2 * self.config['safety_margin']) / 2
        length = (self.config['robot_length'] + 2 * self.config['safety_margin']) / 2
        
        footprint = np.array([
            [-length, -width],
            [length, -width],
            [length, width],
            [-length, width],
            [-length, -width]
        ])
        
        return footprint
    
    def get_effective_radius(self) -> float:
        """Get effective radius including safety margin."""
        return self.config['robot_radius'] + self.config['safety_margin']
    
    def get_planning_resolution(self) -> float:
        """Get grid resolution for planning."""
        return self.config['grid_resolution']
    
    def get_velocity_limits(self) -> Tuple[float, float]:
        """Get velocity limits (linear, angular)."""
        return (self.config['max_linear_velocity'], 
                self.config['max_angular_velocity'])
    
    def get_acceleration_limits(self) -> Tuple[float, float]:
        """Get acceleration limits (linear, angular)."""
        return (self.config['max_linear_acceleration'], 
                self.config['max_angular_acceleration'])
    
    def calculate_stopping_distance(self, current_velocity: float) -> float:
        """Calculate stopping distance for given velocity."""
        if current_velocity <= 0:
            return 0
        
        max_deceleration = self.config['max_linear_acceleration']
        stopping_distance = (current_velocity ** 2) / (2 * max_deceleration)
        
        # Add safety margin
        return stopping_distance + self.config['emergency_stop_distance']
    
    def calculate_trajectory_update_interval(self, robot_speed: float) -> float:
        """Calculate how often to update trajectory based on robot speed."""
        base_update_rate = self.config['trajectory_update_rate']
        
        # Increase update rate for higher speeds
        speed_factor = max(1.0, robot_speed / self.config['max_linear_velocity'])
        adjusted_rate = base_update_rate * speed_factor
        
        return 1.0 / adjusted_rate  # Return interval in seconds
    
    def is_safe_distance(self, distance_to_obstacle: float) -> bool:
        """Check if distance to obstacle is safe."""
        return distance_to_obstacle >= self.config['min_obstacle_distance']
    
    def requires_emergency_stop(self, distance_to_obstacle: float) -> bool:
        """Check if emergency stop is required."""
        return distance_to_obstacle <= self.config['emergency_stop_distance']
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()
    
    def save_config(self, filepath: str):
        """Save configuration to file."""
        import json
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info(f"Configuration saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
    
    def load_config(self, filepath: str):
        """Load configuration from file."""
        import json
        
        try:
            with open(filepath, 'r') as f:
                loaded_config = json.load(f)
            
            self.config.update(loaded_config)
            self._validate_config()
            self.logger.info(f"Configuration loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")

def create_robot_config(custom_params: Dict[str, Any] = None) -> RobotConfig:
    """
    Create a robot configuration with default 3m width.
    
    Args:
        custom_params: Custom parameters to override defaults
        
    Returns:
        RobotConfig instance with 3m default width
    """
    config = {}
    
    if custom_params:
        config.update(custom_params)
    
    return RobotConfig(config)