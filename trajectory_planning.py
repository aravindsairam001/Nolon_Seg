#!/usr/bin/env python3
"""
Floor Segmentation and Trajectory Planning for First-Person Videos

This script combines floor segmentation with obstacle detection and A* path planning
to generate safe robot trajectories from first-person perspective videos.
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
import time

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
from utils.trajectory_pipeline import TrajectoryPlanningPipeline
from utils.robot_config import create_robot_config

class TrajectoryPlanningSystem:
    """Main system for floor segmentation and trajectory planning."""
    
    def __init__(self, 
                 sam_model_path: Optional[str] = None, 
                 device: str = "auto",
                 robot_width: float = 3.0):
        """
        Initialize the trajectory planning system.
        
        Args:
            sam_model_path: Path to SAM model checkpoint
            device: Device to run on ('cpu', 'cuda', 'auto')
            robot_width: Robot width in meters (default: 3.0m)
        """
        # Setup logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.device = self._setup_device(device)
        self.sam_predictor = self._load_sam_model(sam_model_path)
        
        # Initialize components
        self.video_processor = VideoProcessor()
        self.floor_detector = FloorDetector()
        self.visualizer = Visualizer()
        
        # Initialize robot configuration with specified width
        custom_params = {'robot_width': robot_width} if robot_width != 3.0 else None
        self.robot_config = create_robot_config(custom_params)
        self.logger.info(f"Using robot width: {robot_width}m")
        
        # Initialize trajectory planning pipeline
        self.trajectory_pipeline = TrajectoryPlanningPipeline(
            robot_config=self.robot_config,
            enable_dynamic_obstacles=True,
            enable_path_smoothing=True
        )
        
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
                     confidence_threshold: float = 0.5,
                     robot_start_position: Optional[Tuple[int, int]] = None) -> None:
        """
        Process a video to generate floor segmentation and trajectories.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save outputs
            frame_skip: Process every nth frame (1 = all frames)
            confidence_threshold: Minimum confidence for floor detection
            robot_start_position: Initial robot position, or None for auto
        """
        self.logger.info(f"Processing video: {video_path}")
        
        # Create output directories
        output_path = Path(output_dir)
        frames_dir = output_path / "frames"
        masks_dir = output_path / "masks"
        trajectories_dir = output_path / "trajectories"
        videos_dir = output_path / "videos"
        analysis_dir = output_path / "analysis"
        
        for dir_path in [frames_dir, masks_dir, trajectories_dir, videos_dir, analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Extract frames from video
        frames = self.video_processor.extract_frames(video_path, frame_skip)
        
        if not frames:
            self.logger.error("No frames extracted from video")
            return
        
        self.logger.info(f"Extracted {len(frames)} frames")
        
        # Process each frame
        trajectory_results = []
        processed_frames = []
        
        for i, frame in enumerate(tqdm(frames, desc="Processing frames")):
            # Step 1: Detect floor regions
            floor_mask = self.detect_floor_in_frame(frame, confidence_threshold)
            
            # Step 2: Determine robot position
            if robot_start_position is None:
                height, width = frame.shape[:2]
                robot_position = (int(height * 0.8), width // 2)  # Bottom center
            else:
                robot_position = robot_start_position
            
            # Step 3: Plan trajectory
            if floor_mask is not None:
                trajectory_result = self.trajectory_pipeline.process_frame(
                    frame=frame,
                    floor_mask=floor_mask,
                    robot_position=robot_position,
                    goal_direction=(1.0, 0.0)  # Move forward (down in image)
                )
                
                trajectory_results.append(trajectory_result)
                
                # Step 4: Create visualizations
                vis_frame = self.visualizer.visualize_trajectory_planning(frame, trajectory_result)
                processed_frames.append(vis_frame)
                
                # Step 5: Save individual results
                self._save_frame_results(
                    i, frame, trajectory_result, frames_dir, masks_dir, 
                    trajectories_dir, analysis_dir
                )
                
            else:
                processed_frames.append(frame)
                trajectory_results.append(None)
        
        # Step 6: Create output videos
        if processed_frames:
            self._create_output_videos(processed_frames, videos_dir, video_path)
        
        # Step 7: Generate analysis reports
        self._generate_analysis_reports(trajectory_results, analysis_dir)
        
        self.logger.info(f"Processing complete! Results saved to: {output_dir}")
    
    def detect_floor_in_frame(self, frame: np.ndarray, confidence_threshold: float) -> Optional[np.ndarray]:
        """Detect floor regions in a single frame."""
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
    
    def _save_frame_results(self, 
                           frame_idx: int, 
                           frame: np.ndarray,
                           trajectory_result: dict,
                           frames_dir: Path,
                           masks_dir: Path,
                           trajectories_dir: Path,
                           analysis_dir: Path):
        """Save individual frame results."""
        # Save original frame
        frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        
        # Save masks
        if trajectory_result:
            floor_mask = trajectory_result.get('floor_mask')
            obstacle_mask = trajectory_result.get('obstacle_mask')
            
            if floor_mask is not None:
                floor_mask_path = masks_dir / f"floor_mask_{frame_idx:06d}.png"
                cv2.imwrite(str(floor_mask_path), (floor_mask * 255).astype(np.uint8))
            
            if obstacle_mask is not None:
                obstacle_mask_path = masks_dir / f"obstacle_mask_{frame_idx:06d}.png"
                cv2.imwrite(str(obstacle_mask_path), (obstacle_mask * 255).astype(np.uint8))
            
            # Save trajectory data
            trajectory = trajectory_result.get('trajectory')
            if trajectory is not None:
                trajectory_path = trajectories_dir / f"trajectory_{frame_idx:06d}.txt"
                with open(trajectory_path, 'w') as f:
                    f.write("# Trajectory waypoints (row, col)\n")
                    for row, col in trajectory:
                        f.write(f"{row},{col}\n")
                
                # Save world coordinates if available
                world_trajectory = trajectory_result.get('world_trajectory')
                if world_trajectory is not None:
                    world_path = trajectories_dir / f"world_trajectory_{frame_idx:06d}.txt"
                    with open(world_path, 'w') as f:
                        f.write("# World trajectory (x, y in meters)\n")
                        for x, y in world_trajectory:
                            f.write(f"{x:.3f},{y:.3f}\n")
    
    def _create_output_videos(self, processed_frames: List[np.ndarray], videos_dir: Path, video_path: str):
        """Create output videos."""
        output_video_path = videos_dir / f"trajectory_planning_{Path(video_path).stem}.mp4"
        self.video_processor.create_video_from_frames(
            processed_frames, 
            str(output_video_path),
            fps=15  # Slower for easier analysis
        )
        self.logger.info(f"Created output video: {output_video_path}")
    
    def _generate_analysis_reports(self, trajectory_results: List[dict], analysis_dir: Path):
        """Generate comprehensive analysis reports."""
        valid_results = [r for r in trajectory_results if r is not None]
        
        if not valid_results:
            self.logger.warning("No valid trajectory results for analysis")
            return
        
        # Overall statistics
        stats = {
            "total_frames": len(trajectory_results),
            "successful_trajectories": len(valid_results),
            "success_rate": len(valid_results) / len(trajectory_results),
            "average_processing_time": np.mean([r.get('processing_time', 0) for r in valid_results]),
            "average_trajectory_length": np.mean([len(r.get('trajectory', [])) for r in valid_results if r.get('trajectory')]),
            "safety_statistics": self._analyze_safety_statistics(valid_results)
        }
        
        # Save statistics
        stats_path = analysis_dir / "trajectory_statistics.txt"
        with open(stats_path, 'w') as f:
            f.write("Trajectory Planning Analysis Report\n")
            f.write("====================================\n\n")
            
            f.write("Overall Statistics:\n")
            f.write(f"Total frames processed: {stats['total_frames']}\n")
            f.write(f"Successful trajectories: {stats['successful_trajectories']}\n")
            f.write(f"Success rate: {stats['success_rate']:.1%}\n")
            f.write(f"Average processing time: {stats['average_processing_time']:.3f}s\n")
            f.write(f"Average trajectory length: {stats['average_trajectory_length']:.1f} waypoints\n\n")
            
            f.write("Safety Analysis:\n")
            safety_stats = stats['safety_statistics']
            f.write(f"Safe trajectories: {safety_stats['safe_count']}/{safety_stats['total']} ({safety_stats['safe_percentage']:.1%})\n")
            f.write(f"High risk trajectories: {safety_stats['high_risk_count']}\n")
            f.write(f"Emergency stops required: {safety_stats['emergency_stops']}\n")
            f.write(f"Average clearance distance: {safety_stats['avg_clearance']:.2f}m\n")
        
        self.logger.info(f"Analysis report saved to: {stats_path}")
    
    def _analyze_safety_statistics(self, results: List[dict]) -> dict:
        """Analyze safety statistics from trajectory results."""
        safe_count = 0
        high_risk_count = 0
        emergency_stops = 0
        clearances = []
        
        for result in results:
            safety_analysis = result.get('safety_analysis', {})
            
            if safety_analysis.get('is_safe', False):
                safe_count += 1
            
            if safety_analysis.get('collision_risk') == 'high':
                high_risk_count += 1
            
            if safety_analysis.get('emergency_stop_required', False):
                emergency_stops += 1
            
            clearance = safety_analysis.get('clear_distance', 0.0)
            if clearance > 0:
                clearances.append(clearance)
        
        return {
            'total': len(results),
            'safe_count': safe_count,
            'safe_percentage': safe_count / len(results) if results else 0,
            'high_risk_count': high_risk_count,
            'emergency_stops': emergency_stops,
            'avg_clearance': np.mean(clearances) if clearances else 0.0
        }

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Floor segmentation and trajectory planning for first-person videos")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("-o", "--output", default="output", help="Output directory")
    parser.add_argument("--sam-model", help="Path to SAM model checkpoint")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to use")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every nth frame")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--robot-width", type=float, default=3.0, help="Robot width in meters (default: 3.0)")
    
    args = parser.parse_args()
    
    # Initialize system
    system = TrajectoryPlanningSystem(
        sam_model_path=args.sam_model,
        device=args.device,
        robot_width=args.robot_width
    )
    
    # Process video
    system.process_video(
        video_path=args.input_video,
        output_dir=args.output,
        frame_skip=args.frame_skip,
        confidence_threshold=args.confidence
    )
    
    print(f"Trajectory planning complete! Results saved to: {args.output}")

if __name__ == "__main__":
    main()