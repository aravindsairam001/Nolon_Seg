#!/usr/bin/env python3
"""
Floor/Pathway Segmentation for First-Person Resort Videos

This script uses the Segment Anything Model (SAM) and computer vision techniques
to segment floors and pathways from first-person perspective videos.
"""

import os
import cv2
import numpy as np
import torch
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from tqdm import tqdm

# SAM imports
try:
    from segment_anything import SamPredictor, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    print("Warning: Segment Anything not installed. Please install with: pip install git+https://github.com/facebookresearch/segment-anything.git")
    SAM_AVAILABLE = False

from utils.video_processor import VideoProcessor
from utils.floor_detector import FloorDetector
from utils.visualizer import Visualizer

class FloorSegmentationPipeline:
    """Main pipeline for floor segmentation in first-person videos."""
    
    def __init__(self, sam_model_path: Optional[str] = None, device: str = "auto"):
        """
        Initialize the floor segmentation pipeline.
        
        Args:
            sam_model_path: Path to SAM model checkpoint
            device: Device to run on ('cpu', 'cuda', 'auto')
        """
        # Setup logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.device = self._setup_device(device)
        self.sam_predictor = self._load_sam_model(sam_model_path)
        self.video_processor = VideoProcessor()
        self.floor_detector = FloorDetector()
        self.visualizer = Visualizer()
        
    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_sam_model(self, model_path: Optional[str]) -> Optional[SamPredictor]:
        """Load SAM model."""
        if not SAM_AVAILABLE:
            self.logger.warning("SAM not available. Using alternative segmentation methods.")
            return None
            
        if model_path is None:
            # Default model paths to check
            model_paths = [
                "models/sam_vit_h_4b8939.pth",
                "models/sam_vit_l_0b3195.pth", 
                "models/sam_vit_b_01ec64.pth"
            ]
            
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            else:
                self.logger.warning("No SAM model found. Please download a model checkpoint.")
                return None
        
        try:
            # Determine model type from filename
            if "vit_h" in model_path:
                model_type = "vit_h"
            elif "vit_l" in model_path:
                model_type = "vit_l"
            else:
                model_type = "vit_b"
                
            sam = sam_model_registry[model_type](checkpoint=model_path)
            sam.to(device=self.device)
            predictor = SamPredictor(sam)
            
            self.logger.info(f"Loaded SAM model: {model_type} on {self.device}")
            return predictor
            
        except Exception as e:
            self.logger.error(f"Failed to load SAM model: {e}")
            return None
    
    def process_video(self, 
                     video_path: str, 
                     output_dir: str,
                     frame_skip: int = 1,
                     confidence_threshold: float = 0.5) -> None:
        """
        Process a video to segment floors/pathways.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save outputs
            frame_skip: Process every nth frame (1 = all frames)
            confidence_threshold: Minimum confidence for floor detection
        """
        self.logger.info(f"Processing video: {video_path}")
        
        # Create output directories
        output_path = Path(output_dir)
        frames_dir = output_path / "frames"
        masks_dir = output_path / "masks"
        videos_dir = output_path / "videos"
        
        for dir_path in [frames_dir, masks_dir, videos_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Extract frames from video
        frames = self.video_processor.extract_frames(video_path, frame_skip)
        
        if not frames:
            self.logger.error("No frames extracted from video")
            return
        
        self.logger.info(f"Extracted {len(frames)} frames")
        
        # Process each frame
        floor_masks = []
        processed_frames = []
        
        for i, frame in enumerate(tqdm(frames, desc="Processing frames")):
            # Detect floor regions
            floor_mask = self.detect_floor_in_frame(frame, confidence_threshold)
            
            if floor_mask is not None:
                floor_masks.append(floor_mask)
                
                # Save frame and mask
                frame_path = frames_dir / f"frame_{i:06d}.jpg"
                mask_path = masks_dir / f"mask_{i:06d}.png"
                
                cv2.imwrite(str(frame_path), frame)
                cv2.imwrite(str(mask_path), (floor_mask * 255).astype(np.uint8))
                
                # Create visualization
                viz_frame = self.visualizer.overlay_mask(frame, floor_mask)
                processed_frames.append(viz_frame)
            else:
                processed_frames.append(frame)
                floor_masks.append(np.zeros((frame.shape[0], frame.shape[1]), dtype=bool))
        
        # Create output video
        if processed_frames:
            output_video_path = videos_dir / f"segmented_{Path(video_path).stem}.mp4"
            self.video_processor.create_video_from_frames(
                processed_frames, 
                str(output_video_path),
                fps=30
            )
            self.logger.info(f"Created output video: {output_video_path}")
        
        # Generate summary statistics
        self._generate_summary(floor_masks, output_path)
    
    def detect_floor_in_frame(self, frame: np.ndarray, confidence_threshold: float) -> Optional[np.ndarray]:
        """
        Detect floor regions in a single frame.
        
        Args:
            frame: Input frame (BGR format)
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            Binary mask of floor regions or None if detection fails
        """
        try:
            # Convert to RGB for SAM
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Use geometric heuristics to identify potential floor regions
            floor_candidates = self.floor_detector.detect_floor_candidates(frame)
            
            if self.sam_predictor is not None and floor_candidates:
                # Use SAM for precise segmentation
                self.sam_predictor.set_image(rgb_frame)
                
                # Generate points for SAM based on floor candidates
                input_points, input_labels = self._generate_sam_prompts(floor_candidates, frame.shape[:2])
                
                if len(input_points) > 0:
                    masks, scores, _ = self.sam_predictor.predict(
                        point_coords=input_points,
                        point_labels=input_labels,
                        multimask_output=True
                    )
                    
                    # Select best mask based on score and geometric constraints
                    best_mask = self._select_best_floor_mask(masks, scores, frame.shape[:2])
                    
                    if best_mask is not None:
                        return best_mask
            
            # Fallback to geometric-only detection
            return self.floor_detector.detect_floor_geometric(frame)
            
        except Exception as e:
            self.logger.error(f"Error in floor detection: {e}")
            return None
    
    def _generate_sam_prompts(self, floor_candidates: List[np.ndarray], frame_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate point prompts for SAM based on floor candidates."""
        points = []
        labels = []
        
        height, width = frame_shape
        
        # Add positive points from floor candidates
        for candidate_mask in floor_candidates:
            # Find centroid of candidate region
            if np.any(candidate_mask):
                y_coords, x_coords = np.where(candidate_mask)
                if len(y_coords) > 0:
                    centroid_y = int(np.mean(y_coords))
                    centroid_x = int(np.mean(x_coords))
                    points.append([centroid_x, centroid_y])
                    labels.append(1)  # Positive point
        
        # Add negative points (sky region, typically upper part of frame)
        sky_y = height // 4
        sky_points = [[width//4, sky_y], [width//2, sky_y], [3*width//4, sky_y]]
        points.extend(sky_points)
        labels.extend([0, 0, 0])  # Negative points
        
        return np.array(points), np.array(labels)
    
    def _select_best_floor_mask(self, masks: np.ndarray, scores: np.ndarray, frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Select the best floor mask from SAM outputs."""
        if len(masks) == 0:
            return None
        
        height, width = frame_shape
        best_mask = None
        best_score = -1
        
        for mask, score in zip(masks, scores):
            # Check if mask is in lower part of frame (typical for floor)
            lower_half_pixels = np.sum(mask[height//2:, :])
            total_pixels = np.sum(mask)
            
            if total_pixels > 0:
                lower_ratio = lower_half_pixels / total_pixels
                
                # Prefer masks that are mostly in lower half and have good SAM score
                combined_score = score * lower_ratio
                
                if combined_score > best_score and lower_ratio > 0.3:
                    best_score = combined_score
                    best_mask = mask
        
        return best_mask
    
    def _generate_summary(self, floor_masks: List[np.ndarray], output_path: Path) -> None:
        """Generate summary statistics about the segmentation."""
        summary = {
            "total_frames": len(floor_masks),
            "frames_with_floor": sum(1 for mask in floor_masks if np.any(mask)),
            "average_floor_coverage": np.mean([np.sum(mask) / mask.size for mask in floor_masks])
        }
        
        summary_path = output_path / "summary.txt"
        with open(summary_path, 'w') as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        
        self.logger.info(f"Summary saved to: {summary_path}")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Segment floors/pathways in first-person videos")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("-o", "--output", default="output", help="Output directory")
    parser.add_argument("--sam-model", help="Path to SAM model checkpoint")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to use")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every nth frame")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = FloorSegmentationPipeline(
        sam_model_path=args.sam_model,
        device=args.device
    )
    
    # Process video
    pipeline.process_video(
        video_path=args.input_video,
        output_dir=args.output,
        frame_skip=args.frame_skip,
        confidence_threshold=args.confidence
    )
    
    print(f"Processing complete! Results saved to: {args.output}")

if __name__ == "__main__":
    main()