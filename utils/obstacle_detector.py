"""
Obstacle detection utilities for trajectory planning.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
from scipy import ndimage
from sklearn.cluster import DBSCAN

class ObstacleDetector:
    """Detects obstacles in video frames for trajectory planning."""
    
    def __init__(self, min_obstacle_area: int = 100, max_obstacle_area: int = 50000):
        """
        Initialize obstacle detector.
        
        Args:
            min_obstacle_area: Minimum area in pixels for valid obstacles
            max_obstacle_area: Maximum area in pixels for valid obstacles
        """
        self.logger = logging.getLogger(__name__)
        self.min_obstacle_area = min_obstacle_area
        self.max_obstacle_area = max_obstacle_area
        
    def detect_obstacles(self, frame: np.ndarray, floor_mask: np.ndarray) -> np.ndarray:
        """
        Detect obstacles in a frame given the floor mask.
        
        Args:
            frame: Input frame (BGR format)
            floor_mask: Binary mask of floor regions
            
        Returns:
            Binary mask of detected obstacles
        """
        try:
            # Method 1: Height-based obstacle detection
            height_obstacles = self._detect_height_based_obstacles(frame, floor_mask)
            
            # Method 2: Edge-based obstacle detection
            edge_obstacles = self._detect_edge_based_obstacles(frame, floor_mask)
            
            # Method 3: Color contrast obstacles
            contrast_obstacles = self._detect_contrast_obstacles(frame, floor_mask)
            
            # Method 4: SAM-based object detection (if available)
            sam_obstacles = self._detect_sam_obstacles(frame, floor_mask)
            
            # Combine all detection methods
            combined_obstacles = self._combine_obstacle_detections([
                height_obstacles,
                edge_obstacles, 
                contrast_obstacles,
                sam_obstacles
            ])
            
            # Filter and clean obstacles
            filtered_obstacles = self._filter_obstacles(combined_obstacles, frame.shape[:2])
            
            return filtered_obstacles
            
        except Exception as e:
            self.logger.error(f"Error in obstacle detection: {e}")
            return np.zeros(frame.shape[:2], dtype=bool)
    
    def _detect_height_based_obstacles(self, frame: np.ndarray, floor_mask: np.ndarray) -> np.ndarray:
        """Detect obstacles based on height assumption in first-person view."""
        height, width = frame.shape[:2]
        
        # In first-person view, obstacles typically:
        # 1. Appear above the floor level
        # 2. Have vertical structure
        # 3. Are darker/different from floor
        
        # Create gradient-based height map
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Vertical gradient (obstacles have strong vertical edges)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_y = np.abs(grad_y)
        
        # Normalize and threshold
        grad_y = (grad_y / grad_y.max() * 255).astype(np.uint8)
        _, height_mask = cv2.threshold(grad_y, 50, 255, cv2.THRESH_BINARY)
        
        # Remove floor regions
        height_mask = height_mask.astype(bool) & ~floor_mask
        
        # Keep only upper regions (where obstacles are likely)
        upper_region = np.zeros_like(height_mask)
        upper_region[:int(height * 0.7), :] = True
        height_mask = height_mask & upper_region
        
        return height_mask
    
    def _detect_edge_based_obstacles(self, frame: np.ndarray, floor_mask: np.ndarray) -> np.ndarray:
        """Detect obstacles using edge information."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Multi-scale edge detection
        edges1 = cv2.Canny(gray, 50, 150)
        edges2 = cv2.Canny(gray, 30, 100)
        
        # Combine edges
        combined_edges = cv2.bitwise_or(edges1, edges2)
        
        # Dilate to create obstacle regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edge_obstacles = cv2.dilate(combined_edges, kernel, iterations=2)
        
        # Remove floor regions
        edge_obstacles = edge_obstacles.astype(bool) & ~floor_mask
        
        return edge_obstacles
    
    def _detect_contrast_obstacles(self, frame: np.ndarray, floor_mask: np.ndarray) -> np.ndarray:
        """Detect obstacles based on color/texture contrast with floor."""
        # Get average floor color
        floor_pixels = frame[floor_mask]
        if len(floor_pixels) == 0:
            return np.zeros(frame.shape[:2], dtype=bool)
        
        avg_floor_color = np.mean(floor_pixels, axis=0)
        
        # Calculate color distance from floor
        color_diff = np.linalg.norm(frame - avg_floor_color, axis=2)
        
        # Threshold for obstacles (significantly different from floor)
        threshold = np.percentile(color_diff, 75)  # Top 25% different regions
        contrast_mask = color_diff > threshold
        
        # Remove floor regions
        contrast_obstacles = contrast_mask & ~floor_mask
        
        return contrast_obstacles
    
    def _detect_sam_obstacles(self, frame: np.ndarray, floor_mask: np.ndarray) -> np.ndarray:
        """Detect obstacles using SAM if available."""
        # This is a placeholder for SAM-based object detection
        # Could be enhanced to use SAM for precise object segmentation
        height, width = frame.shape[:2]
        
        # For now, return empty mask
        # TODO: Integrate with SAM for object detection
        return np.zeros((height, width), dtype=bool)
    
    def _combine_obstacle_detections(self, obstacle_masks: List[np.ndarray]) -> np.ndarray:
        """Combine multiple obstacle detection methods."""
        if not obstacle_masks:
            return np.zeros((100, 100), dtype=bool)
        
        # Filter out None masks
        valid_masks = [mask for mask in obstacle_masks if mask is not None and mask.size > 0]
        
        if not valid_masks:
            return np.zeros_like(obstacle_masks[0])
        
        # Voting-based combination
        combined = np.zeros_like(valid_masks[0], dtype=int)
        
        for mask in valid_masks:
            combined += mask.astype(int)
        
        # Require at least 2 methods to agree
        threshold = min(2, len(valid_masks))
        final_mask = combined >= threshold
        
        return final_mask
    
    def _filter_obstacles(self, obstacle_mask: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Filter and clean detected obstacles."""
        if not np.any(obstacle_mask):
            return obstacle_mask
        
        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(obstacle_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        
        # Remove very small and very large components
        labeled, num_features = ndimage.label(cleaned)
        
        if num_features == 0:
            return obstacle_mask.astype(bool)
        
        # Calculate component sizes
        component_sizes = ndimage.sum(cleaned, labeled, range(1, num_features + 1))
        
        # Keep only appropriately sized components
        valid_components = np.where(
            (component_sizes >= self.min_obstacle_area) & 
            (component_sizes <= self.max_obstacle_area)
        )[0] + 1
        
        filtered_mask = np.zeros_like(obstacle_mask)
        for component_id in valid_components:
            filtered_mask[labeled == component_id] = True
        
        return filtered_mask.astype(bool)
    
    def get_obstacle_contours(self, obstacle_mask: np.ndarray) -> List[np.ndarray]:
        """Get contours of detected obstacles."""
        contours, _ = cv2.findContours(
            obstacle_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by area
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_obstacle_area <= area <= self.max_obstacle_area:
                valid_contours.append(contour)
        
        return valid_contours
    
    def get_obstacle_bounding_boxes(self, obstacle_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Get bounding boxes of detected obstacles."""
        contours = self.get_obstacle_contours(obstacle_mask)
        bounding_boxes = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))
        
        return bounding_boxes
    
    def create_obstacle_distance_map(self, obstacle_mask: np.ndarray) -> np.ndarray:
        """Create distance map from obstacles for path planning."""
        # Invert mask for distance transform
        free_space = ~obstacle_mask
        
        # Calculate distance transform
        distance_map = cv2.distanceTransform(
            free_space.astype(np.uint8), 
            cv2.DIST_L2, 
            5
        )
        
        return distance_map
    
    def detect_dynamic_obstacles(self, 
                                current_frame: np.ndarray, 
                                previous_frame: Optional[np.ndarray],
                                floor_mask: np.ndarray) -> np.ndarray:
        """Detect moving obstacles by comparing frames."""
        if previous_frame is None:
            return np.zeros(current_frame.shape[:2], dtype=bool)
        
        # Calculate frame difference
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        
        frame_diff = cv2.absdiff(gray_current, gray_previous)
        
        # Threshold to find moving regions
        _, motion_mask = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        
        # Only consider motion outside floor regions
        dynamic_obstacles = motion_mask.astype(bool) & ~floor_mask
        
        return dynamic_obstacles