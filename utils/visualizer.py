"""
Visualization utilities for floor segmentation results and trajectory planning.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
import logging

class Visualizer:
    """Handles visualization of segmentation results, trajectories, and analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Color schemes for visualization
        self.floor_color = (0, 255, 0)     # Green for floor
        self.obstacle_color = (0, 0, 255)  # Red for obstacles
        self.trajectory_color = (255, 255, 0)  # Cyan for trajectory
        self.robot_color = (255, 0, 255)   # Magenta for robot
        self.dynamic_obstacle_color = (0, 165, 255)  # Orange for dynamic obstacles
        self.overlay_alpha = 0.4
        
    def visualize_trajectory_planning(self,
                                    frame: np.ndarray,
                                    results: Dict[str, Any]) -> np.ndarray:
        """
        Create comprehensive visualization of trajectory planning results.
        
        Args:
            frame: Original frame
            results: Results from trajectory planning pipeline
            
        Returns:
            Visualization image with all components
        """
        vis_frame = frame.copy()
        
        # Extract results
        floor_mask = results.get('floor_mask')
        obstacle_mask = results.get('obstacle_mask') 
        dynamic_obstacle_mask = results.get('dynamic_obstacle_mask')
        trajectory = results.get('trajectory')
        robot_position = results.get('robot_position')
        safety_analysis = results.get('safety_analysis', {})
        
        # Step 1: Overlay floor mask
        if floor_mask is not None:
            vis_frame = self.overlay_mask(vis_frame, floor_mask, self.floor_color, 0.2)
        
        # Step 2: Overlay static obstacles
        if obstacle_mask is not None:
            vis_frame = self.overlay_mask(vis_frame, obstacle_mask, self.obstacle_color, 0.6)
        
        # Step 3: Overlay dynamic obstacles
        if dynamic_obstacle_mask is not None and np.any(dynamic_obstacle_mask):
            vis_frame = self.overlay_mask(vis_frame, dynamic_obstacle_mask, 
                                        self.dynamic_obstacle_color, 0.6)
        
        # Step 4: Draw trajectory
        if trajectory is not None:
            vis_frame = self.draw_trajectory(vis_frame, trajectory, safety_analysis)
        
        # Step 5: Draw robot position and footprint
        if robot_position is not None:
            vis_frame = self.draw_robot(vis_frame, robot_position)
        
        # Step 6: Add safety information
        vis_frame = self.add_safety_overlay(vis_frame, safety_analysis)
        
        return vis_frame
    
    def draw_trajectory(self, 
                       frame: np.ndarray, 
                       trajectory: List[Tuple[int, int]],
                       safety_analysis: Dict[str, Any] = None) -> np.ndarray:
        """Draw trajectory path on frame."""
        if not trajectory or len(trajectory) < 2:
            return frame
        
        result = frame.copy()
        
        # Determine trajectory color based on safety
        if safety_analysis:
            risk_level = safety_analysis.get('collision_risk', 'unknown')
            if risk_level == 'high':
                color = (0, 0, 255)  # Red for high risk
            elif risk_level == 'medium':
                color = (0, 165, 255)  # Orange for medium risk
            else:
                color = self.trajectory_color  # Cyan for low risk
        else:
            color = self.trajectory_color
        
        # Draw trajectory line
        points = np.array([(col, row) for row, col in trajectory], dtype=np.int32)
        cv2.polylines(result, [points], False, color, 3)
        
        # Draw waypoints
        for i, (row, col) in enumerate(trajectory):
            if i == 0:
                # Start point
                cv2.circle(result, (col, row), 8, (0, 255, 0), -1)
                cv2.circle(result, (col, row), 8, (0, 0, 0), 2)
            elif i == len(trajectory) - 1:
                # End point
                cv2.circle(result, (col, row), 8, (255, 0, 0), -1)
                cv2.circle(result, (col, row), 8, (0, 0, 0), 2)
            else:
                # Intermediate waypoints
                cv2.circle(result, (col, row), 4, color, -1)
        
        # Add trajectory direction arrows
        if len(trajectory) >= 2:
            result = self.add_direction_arrows(result, trajectory, color)
        
        return result
    
    def add_direction_arrows(self, 
                           frame: np.ndarray, 
                           trajectory: List[Tuple[int, int]], 
                           color: Tuple[int, int, int]) -> np.ndarray:
        """Add direction arrows along trajectory."""
        result = frame.copy()
        
        # Add arrows every few waypoints
        arrow_spacing = max(1, len(trajectory) // 5)
        
        for i in range(0, len(trajectory) - 1, arrow_spacing):
            if i + 1 < len(trajectory):
                start_point = trajectory[i]
                end_point = trajectory[i + 1]
                
                # Convert to (x, y) format for cv2
                start = (start_point[1], start_point[0])
                end = (end_point[1], end_point[0])
                
                # Draw arrow
                cv2.arrowedLine(result, start, end, color, 2, tipLength=0.3)
        
        return result
    
    def draw_robot(self, 
                  frame: np.ndarray, 
                  robot_position: Tuple[int, int],
                  robot_heading: float = 0.0,
                  robot_radius: float = 20) -> np.ndarray:
        """Draw robot position and orientation."""
        result = frame.copy()
        row, col = robot_position
        center = (col, row)
        
        # Draw robot body (circle)
        cv2.circle(result, center, int(robot_radius), self.robot_color, -1)
        cv2.circle(result, center, int(robot_radius), (0, 0, 0), 2)
        
        # Draw robot heading direction
        end_x = col + int(robot_radius * 0.8 * np.cos(robot_heading))
        end_y = row + int(robot_radius * 0.8 * np.sin(robot_heading))
        cv2.arrowedLine(result, center, (end_x, end_y), (255, 255, 255), 3)
        
        # Add robot label
        cv2.putText(result, "ROBOT", (col - 20, row - robot_radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result
    
    def add_safety_overlay(self, 
                          frame: np.ndarray, 
                          safety_analysis: Dict[str, Any]) -> np.ndarray:
        """Add safety information overlay."""
        if not safety_analysis:
            return frame
        
        result = frame.copy()
        height, width = frame.shape[:2]
        
        # Create overlay area
        overlay_height = 120
        overlay_area = np.zeros((overlay_height, width, 3), dtype=np.uint8)
        
        # Background color based on safety status
        is_safe = safety_analysis.get('is_safe', False)
        emergency_stop = safety_analysis.get('emergency_stop_required', False)
        
        if emergency_stop:
            bg_color = (0, 0, 139)  # Dark red
        elif not is_safe:
            bg_color = (0, 100, 139)  # Dark orange
        else:
            bg_color = (0, 100, 0)  # Dark green
        
        overlay_area[:] = bg_color
        
        # Add safety information text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (255, 255, 255)
        thickness = 1
        
        y_offset = 20
        line_spacing = 25
        
        # Status line
        status = "EMERGENCY STOP" if emergency_stop else ("UNSAFE" if not is_safe else "SAFE")
        cv2.putText(overlay_area, f"Status: {status}", (10, y_offset), 
                   font, font_scale, font_color, thickness)
        
        # Risk level
        risk = safety_analysis.get('collision_risk', 'unknown').upper()
        cv2.putText(overlay_area, f"Risk: {risk}", (10, y_offset + line_spacing), 
                   font, font_scale, font_color, thickness)
        
        # Clearance distance
        clearance = safety_analysis.get('clear_distance', 0.0)
        cv2.putText(overlay_area, f"Clearance: {clearance:.2f}m", (10, y_offset + 2*line_spacing), 
                   font, font_scale, font_color, thickness)
        
        # Confidence
        confidence = safety_analysis.get('confidence', 0.0)
        cv2.putText(overlay_area, f"Confidence: {confidence:.1%}", (10, y_offset + 3*line_spacing), 
                   font, font_scale, font_color, thickness)
        
        # Add overlay to frame
        result[height-overlay_height:height, :] = overlay_area
        
        return result
    
    def create_trajectory_metrics_visualization(self, 
                                              metrics: Dict[str, Any]) -> np.ndarray:
        """Create visualization of trajectory quality metrics."""
        # Create metrics display
        img_width, img_height = 400, 300
        metrics_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)
        thickness = 1
        
        y_start = 30
        line_spacing = 25
        
        # Title
        cv2.putText(metrics_img, "Trajectory Metrics", (10, y_start), 
                   font, 0.7, (0, 255, 255), 2)
        
        y_pos = y_start + 30
        
        # Display metrics
        metric_names = {
            'length': 'Length (m)',
            'straightness': 'Straightness',
            'smoothness': 'Smoothness', 
            'obstacle_clearance': 'Avg Clearance (m)',
            'execution_feasibility': 'Feasibility',
            'overall_quality': 'Overall Quality'
        }
        
        for key, display_name in metric_names.items():
            if key in metrics:
                value = metrics[key]
                if key in ['length', 'obstacle_clearance']:
                    text = f"{display_name}: {value:.2f}"
                else:
                    text = f"{display_name}: {value:.2f}"
                
                # Color code based on value
                if key == 'overall_quality':
                    if value > 0.7:
                        color = (0, 255, 0)  # Green
                    elif value > 0.4:
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (0, 0, 255)  # Red
                else:
                    color = font_color
                
                cv2.putText(metrics_img, text, (10, y_pos), 
                           font, font_scale, color, thickness)
                y_pos += line_spacing
        
        return metrics_img
    
    def create_obstacle_analysis_visualization(self,
                                             frame: np.ndarray,
                                             obstacle_mask: np.ndarray,
                                             distance_map: Optional[np.ndarray] = None) -> np.ndarray:
        """Create visualization showing obstacle analysis."""
        if distance_map is None:
            # Create distance map if not provided
            free_space = ~obstacle_mask
            distance_map = cv2.distanceTransform(
                free_space.astype(np.uint8), cv2.DIST_L2, 5
            )
        
        # Create heatmap of distances
        distance_normalized = cv2.normalize(distance_map, None, 0, 255, cv2.NORM_MINMAX)
        distance_colored = cv2.applyColorMap(distance_normalized.astype(np.uint8), 
                                           cv2.COLORMAP_JET)
        
        # Blend with original frame
        result = cv2.addWeighted(frame, 0.6, distance_colored, 0.4, 0)
        
        # Overlay obstacles in red
        result = self.overlay_mask(result, obstacle_mask, (0, 0, 255), 0.8)
        
        # Add colorbar legend
        cv2.putText(result, "Distance to Obstacles", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result, "Blue: Far", (10, result.shape[0] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(result, "Red: Near", (10, result.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return result
        
    def overlay_mask(self, 
                    frame: np.ndarray, 
                    mask: np.ndarray, 
                    color: Optional[Tuple[int, int, int]] = None,
                    alpha: float = None) -> np.ndarray:
        """
        Overlay a binary mask on a frame with specified color.
        
        Args:
            frame: Input frame (BGR format)
            mask: Binary mask to overlay
            color: RGB color for overlay (default: green)
            alpha: Transparency factor (default: 0.4)
            
        Returns:
            Frame with overlaid mask
        """
        if color is None:
            color = self.floor_color
        if alpha is None:
            alpha = self.overlay_alpha
            
        result = frame.copy()
        
        # Create colored overlay
        overlay = np.zeros_like(frame)
        overlay[mask] = color
        
        # Blend with original frame
        result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)
        
        return result
    
    def overlay_multiple_masks(self, 
                             frame: np.ndarray, 
                             masks: List[np.ndarray], 
                             colors: Optional[List[Tuple[int, int, int]]] = None,
                             alpha: float = None) -> np.ndarray:
        """
        Overlay multiple masks with different colors.
        
        Args:
            frame: Input frame (BGR format)
            masks: List of binary masks
            colors: List of RGB colors for each mask
            alpha: Transparency factor
            
        Returns:
            Frame with overlaid masks
        """
        if alpha is None:
            alpha = self.overlay_alpha
            
        if colors is None:
            # Default color palette
            colors = [
                (0, 255, 0),    # Green
                (255, 0, 0),    # Red
                (0, 0, 255),    # Blue
                (255, 255, 0),  # Yellow
                (255, 0, 255),  # Magenta
            ]
        
        result = frame.copy()
        
        for i, mask in enumerate(masks):
            if np.any(mask):
                color = colors[i % len(colors)]
                overlay = np.zeros_like(frame)
                overlay[mask] = color
                result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)
        
        return result
    
    def create_side_by_side(self, 
                          original: np.ndarray, 
                          segmented: np.ndarray,
                          title1: str = "Original",
                          title2: str = "Segmented") -> np.ndarray:
        """
        Create side-by-side comparison of original and segmented frames.
        
        Args:
            original: Original frame
            segmented: Segmented frame
            title1: Title for original frame
            title2: Title for segmented frame
            
        Returns:
            Combined side-by-side image
        """
        # Ensure both images have the same height
        h1, w1 = original.shape[:2]
        h2, w2 = segmented.shape[:2]
        
        target_height = max(h1, h2)
        
        # Resize if necessary
        if h1 != target_height:
            original = cv2.resize(original, (int(w1 * target_height / h1), target_height))
        if h2 != target_height:
            segmented = cv2.resize(segmented, (int(w2 * target_height / h2), target_height))
        
        # Create combined image
        combined = np.hstack([original, segmented])
        
        # Add titles
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        color = (255, 255, 255)
        
        cv2.putText(combined, title1, (10, 30), font, font_scale, color, thickness)
        cv2.putText(combined, title2, (original.shape[1] + 10, 30), font, font_scale, color, thickness)
        
        return combined
    
    def create_mask_visualization(self, mask: np.ndarray, title: str = "Mask") -> np.ndarray:
        """
        Create a visualization of a binary mask.
        
        Args:
            mask: Binary mask
            title: Title for the visualization
            
        Returns:
            Colored visualization of the mask
        """
        # Convert boolean mask to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Create colored version
        colored_mask = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_GREEN)
        
        # Add title
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(colored_mask, title, (10, 30), font, 1, (255, 255, 255), 2)
        
        return colored_mask
    
    def create_progress_visualization(self, 
                                   frames: List[np.ndarray], 
                                   masks: List[np.ndarray],
                                   frame_indices: List[int]) -> np.ndarray:
        """
        Create a progress visualization showing multiple frames with masks.
        
        Args:
            frames: List of frames
            masks: List of corresponding masks
            frame_indices: Frame indices for labeling
            
        Returns:
            Grid visualization of frames with masks
        """
        if not frames or len(frames) != len(masks):
            self.logger.error("Frames and masks lists must have same length")
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Determine grid layout
        n_frames = len(frames)
        cols = min(4, n_frames)
        rows = (n_frames + cols - 1) // cols
        
        # Resize frames to consistent size
        target_size = (200, 150)  # width, height
        resized_frames = []
        
        for frame, mask in zip(frames, masks):
            # Overlay mask on frame
            overlaid = self.overlay_mask(frame, mask)
            
            # Resize
            resized = cv2.resize(overlaid, target_size)
            resized_frames.append(resized)
        
        # Create grid
        grid_height = rows * target_size[1] + (rows - 1) * 10  # 10px spacing
        grid_width = cols * target_size[0] + (cols - 1) * 10
        grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        for i, (frame, frame_idx) in enumerate(zip(resized_frames, frame_indices)):
            row = i // cols
            col = i % cols
            
            y_start = row * (target_size[1] + 10)
            y_end = y_start + target_size[1]
            x_start = col * (target_size[0] + 10)
            x_end = x_start + target_size[0]
            
            grid[y_start:y_end, x_start:x_end] = frame
            
            # Add frame number
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(grid, f"Frame {frame_idx}", 
                       (x_start + 5, y_start + 20), font, 0.5, (255, 255, 255), 1)
        
        return grid
    
    def plot_segmentation_statistics(self, 
                                   floor_masks: List[np.ndarray], 
                                   save_path: Optional[str] = None) -> None:
        """
        Create statistical plots about the segmentation results.
        
        Args:
            floor_masks: List of floor masks
            save_path: Optional path to save the plot
        """
        if not floor_masks:
            self.logger.error("No masks provided for statistics")
            return
        
        # Calculate statistics
        frame_numbers = list(range(len(floor_masks)))
        floor_coverage = []
        mask_areas = []
        
        for mask in floor_masks:
            total_pixels = mask.size
            floor_pixels = np.sum(mask)
            coverage = floor_pixels / total_pixels if total_pixels > 0 else 0
            
            floor_coverage.append(coverage * 100)  # Convert to percentage
            mask_areas.append(floor_pixels)
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: Floor coverage over time
        ax1.plot(frame_numbers, floor_coverage, 'b-', linewidth=2)
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Floor Coverage (%)')
        ax1.set_title('Floor Coverage Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Histogram of floor coverage
        ax2.hist(floor_coverage, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Floor Coverage (%)')
        ax2.set_ylabel('Number of Frames')
        ax2.set_title('Distribution of Floor Coverage')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Mask area over time
        ax3.plot(frame_numbers, mask_areas, 'r-', linewidth=2)
        ax3.set_xlabel('Frame Number')
        ax3.set_ylabel('Floor Area (pixels)')
        ax3.set_title('Floor Area Over Time')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        stats_text = f"""
        Total Frames: {len(floor_masks)}
        Frames with Floor: {sum(1 for mask in floor_masks if np.any(mask))}
        Avg Coverage: {np.mean(floor_coverage):.1f}%
        Max Coverage: {np.max(floor_coverage):.1f}%
        Min Coverage: {np.min(floor_coverage):.1f}%
        Std Coverage: {np.std(floor_coverage):.1f}%
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.axis('off')
        ax4.set_title('Summary Statistics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Statistics plot saved to: {save_path}")
        
        plt.show()
    
    def create_confidence_heatmap(self, 
                                frame: np.ndarray, 
                                confidence_map: np.ndarray,
                                title: str = "Confidence Map") -> np.ndarray:
        """
        Create a heatmap visualization of confidence scores.
        
        Args:
            frame: Original frame
            confidence_map: 2D array of confidence scores (0-1)
            title: Title for the visualization
            
        Returns:
            Heatmap visualization
        """
        # Normalize confidence map to 0-255
        conf_normalized = (confidence_map * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(conf_normalized, cv2.COLORMAP_JET)
        
        # Blend with original frame
        blended = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
        
        # Add title and colorbar legend
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(blended, title, (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(blended, "Low", (10, blended.shape[0] - 40), font, 0.6, (255, 0, 0), 2)
        cv2.putText(blended, "High", (10, blended.shape[0] - 10), font, 0.6, (0, 0, 255), 2)
        
        return blended