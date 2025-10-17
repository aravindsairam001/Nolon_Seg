"""
Floor detection utilities using geometric heuristics and computer vision techniques.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
from sklearn.cluster import KMeans
from scipy import ndimage

class FloorDetector:
    """Detects floor/pathway regions using geometric and color-based heuristics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_floor_candidates(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Detect potential floor regions using multiple heuristics.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of binary masks representing potential floor regions
        """
        candidates = []
        
        # Method 1: Color-based segmentation
        color_mask = self._detect_floor_by_color(frame)
        if np.any(color_mask):
            candidates.append(color_mask)
        
        # Method 2: Texture-based detection
        texture_mask = self._detect_floor_by_texture(frame)
        if np.any(texture_mask):
            candidates.append(texture_mask)
        
        # Method 3: Geometric constraints (lower part of image)
        geometric_mask = self._detect_floor_by_geometry(frame)
        if np.any(geometric_mask):
            candidates.append(geometric_mask)
        
        # Method 4: Edge-based detection
        edge_mask = self._detect_floor_by_edges(frame)
        if np.any(edge_mask):
            candidates.append(edge_mask)
        
        return candidates
    
    def detect_floor_geometric(self, frame: np.ndarray) -> np.ndarray:
        """
        Fallback floor detection using only geometric constraints.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Binary mask of detected floor region
        """
        height, width = frame.shape[:2]
        
        # Start with lower portion of image
        mask = np.zeros((height, width), dtype=bool)
        
        # Assume floor is in bottom 60% of image
        floor_start_y = int(height * 0.4)
        mask[floor_start_y:, :] = True
        
        # Refine using color consistency
        lower_region = frame[floor_start_y:, :]
        if lower_region.size > 0:
            # Get dominant color in lower region
            dominant_color = self._get_dominant_color(lower_region)
            
            # Create color-based mask
            color_tolerance = 50
            color_mask = self._create_color_mask(frame, dominant_color, color_tolerance)
            
            # Combine with geometric constraint
            mask = mask & color_mask
        
        return mask
    
    def _detect_floor_by_color(self, frame: np.ndarray) -> np.ndarray:
        """Detect floor using color clustering and spatial constraints."""
        height, width = frame.shape[:2]
        
        # Focus on lower half of image for color analysis
        lower_half = frame[height//2:, :]
        
        # Convert to LAB color space for better color clustering
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab_lower = lab_frame[height//2:, :]
        
        # Reshape for clustering
        pixels = lab_lower.reshape(-1, 3)
        
        if len(pixels) == 0:
            return np.zeros((height, width), dtype=bool)
        
        # Perform color clustering
        n_clusters = min(5, len(np.unique(pixels.view(np.void), axis=0)))
        if n_clusters < 2:
            return np.zeros((height, width), dtype=bool)
        
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pixels)
            
            # Find the most common cluster (likely floor)
            unique_labels, counts = np.unique(labels, return_counts=True)
            dominant_cluster = unique_labels[np.argmax(counts)]
            
            # Create mask for dominant cluster
            cluster_mask = (labels == dominant_cluster).reshape(lab_lower.shape[:2])
            
            # Extend mask to full image
            full_mask = np.zeros((height, width), dtype=bool)
            full_mask[height//2:, :] = cluster_mask
            
            # Apply spatial filtering to ensure floor-like characteristics
            full_mask = self._apply_spatial_filtering(full_mask, frame.shape[:2])
            
            return full_mask
            
        except Exception as e:
            self.logger.warning(f"Color clustering failed: {e}")
            return np.zeros((height, width), dtype=bool)
    
    def _detect_floor_by_texture(self, frame: np.ndarray) -> np.ndarray:
        """Detect floor using texture analysis."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Calculate local standard deviation (texture measure)
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        
        # Mean filter
        mean_filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        
        # Standard deviation
        sqr_diff = (gray.astype(np.float32) - mean_filtered) ** 2
        std_dev_squared = cv2.filter2D(sqr_diff, -1, kernel)
        # Ensure non-negative values before taking square root
        std_dev_squared = np.maximum(std_dev_squared, 0)
        std_dev = np.sqrt(std_dev_squared)
        
        # Floor typically has low to moderate texture
        # Too low = sky/walls, too high = complex objects
        texture_mask = (std_dev > 5) & (std_dev < 30)
        
        # Apply geometric constraint (floor is typically in lower part)
        geometric_weight = np.linspace(0, 1, height).reshape(-1, 1)
        geometric_weight = np.tile(geometric_weight, (1, width))
        
        # Combine texture and geometric information
        floor_score = texture_mask.astype(float) * geometric_weight
        
        # Threshold to create binary mask
        threshold = np.percentile(floor_score[floor_score > 0], 70) if np.any(floor_score > 0) else 0.5
        mask = floor_score > threshold
        
        return mask
    
    def _detect_floor_by_geometry(self, frame: np.ndarray) -> np.ndarray:
        """Detect floor using geometric perspective constraints."""
        height, width = frame.shape[:2]
        
        # Create a gradient mask that favors the bottom of the image
        y_coords = np.arange(height).reshape(-1, 1)
        x_coords = np.arange(width).reshape(1, -1)
        
        # Linear gradient from top to bottom
        gradient = y_coords / height
        gradient = np.tile(gradient, (1, width))
        
        # Apply perspective correction - floor appears larger at bottom
        # and converges toward vanishing point
        center_x = width // 2
        x_distance = np.abs(x_coords - center_x) / (width // 2)
        
        # Combine vertical position and horizontal distance from center
        perspective_score = gradient * (1 - 0.3 * x_distance)
        
        # Create mask based on score threshold
        threshold = 0.6  # Prefer bottom 40% of image
        mask = perspective_score > threshold
        
        return mask
    
    def _detect_floor_by_edges(self, frame: np.ndarray) -> np.ndarray:
        """Detect floor using edge information and line detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return np.zeros((height, width), dtype=bool)
        
        # Create mask for horizontal-ish lines (floor edges)
        mask = np.zeros((height, width), dtype=bool)
        
        for line in lines:
            rho, theta = line[0]
            
            # Focus on nearly horizontal lines (floor boundaries)
            angle_deg = np.degrees(theta)
            if abs(angle_deg - 90) < 30 or abs(angle_deg - 270) < 30:  # Within 30 degrees of horizontal
                # Convert to Cartesian coordinates
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                
                # Calculate line endpoints
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                
                # Draw line on mask
                cv2.line(mask.astype(np.uint8), (x1, y1), (x2, y2), 1, 2)
        
        # Dilate to create regions below horizontal lines
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        
        # Fill areas below detected lines
        for y in range(height-1, -1, -1):
            if np.any(mask[y, :]):
                mask[y:, :] = mask[y:, :] | mask[y, :]
                break
        
        return mask
    
    def _get_dominant_color(self, region: np.ndarray) -> np.ndarray:
        """Get dominant color in a region using K-means clustering."""
        if region.size == 0:
            return np.array([0, 0, 0])
        
        # Reshape to list of pixels
        pixels = region.reshape(-1, 3)
        
        # Remove very dark pixels (shadows)
        brightness = np.mean(pixels, axis=1)
        bright_pixels = pixels[brightness > 30]
        
        if len(bright_pixels) == 0:
            return np.mean(pixels, axis=0)
        
        try:
            # Cluster colors
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(bright_pixels)
            
            # Return the most common cluster center
            labels = kmeans.labels_
            unique_labels, counts = np.unique(labels, return_counts=True)
            dominant_cluster = unique_labels[np.argmax(counts)]
            
            return kmeans.cluster_centers_[dominant_cluster]
        except:
            return np.mean(bright_pixels, axis=0)
    
    def _create_color_mask(self, frame: np.ndarray, target_color: np.ndarray, tolerance: int) -> np.ndarray:
        """Create a mask based on color similarity."""
        # Calculate color distance
        color_diff = np.linalg.norm(frame - target_color, axis=2)
        mask = color_diff < tolerance
        
        return mask
    
    def _apply_spatial_filtering(self, mask: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Apply spatial filtering to ensure floor-like characteristics."""
        height, width = frame_shape
        
        # Remove small disconnected components
        labeled_mask, num_features = ndimage.label(mask)
        
        if num_features == 0:
            return mask
        
        # Keep only large connected components
        component_sizes = ndimage.sum(mask, labeled_mask, range(1, num_features + 1))
        min_size = (height * width) * 0.05  # At least 5% of image
        
        large_components = np.where(component_sizes > min_size)[0] + 1
        
        filtered_mask = np.zeros_like(mask)
        for component_id in large_components:
            filtered_mask[labeled_mask == component_id] = True
        
        # Apply morphological operations to smooth the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        filtered_mask = cv2.morphologyEx(filtered_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, kernel)
        
        return filtered_mask.astype(bool)