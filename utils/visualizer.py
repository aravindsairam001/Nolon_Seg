"""
Visualization utilities for floor segmentation results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import logging

class Visualizer:
    """Handles visualization of segmentation results and analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Color schemes for visualization
        self.floor_color = (0, 255, 0)  # Green for floor
        self.overlay_alpha = 0.4
        
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