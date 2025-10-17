"""
Video processing utilities for floor segmentation pipeline.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import logging

class VideoProcessor:
    """Handles video input/output operations and frame processing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_frames(self, video_path: str, frame_skip: int = 1) -> List[np.ndarray]:
        """
        Extract frames from video file.
        
        Args:
            video_path: Path to video file
            frame_skip: Extract every nth frame (1 = all frames)
            
        Returns:
            List of frames as numpy arrays (BGR format)
        """
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                self.logger.error(f"Could not open video: {video_path}")
                return frames
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"Video info: {frame_count} frames, {fps:.2f} fps")
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Skip frames based on frame_skip parameter
                if frame_idx % frame_skip == 0:
                    frames.append(frame.copy())
                
                frame_idx += 1
            
            cap.release()
            
        except Exception as e:
            self.logger.error(f"Error extracting frames: {e}")
        
        return frames
    
    def create_video_from_frames(self, 
                               frames: List[np.ndarray], 
                               output_path: str, 
                               fps: float = 30.0,
                               codec: str = 'mp4v') -> bool:
        """
        Create video from list of frames.
        
        Args:
            frames: List of frames as numpy arrays
            output_path: Path for output video
            fps: Frames per second for output video
            codec: Video codec to use
            
        Returns:
            True if successful, False otherwise
        """
        if not frames:
            self.logger.error("No frames provided for video creation")
            return False
        
        try:
            # Get frame dimensions
            height, width = frames[0].shape[:2]
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                self.logger.error(f"Could not open video writer for: {output_path}")
                return False
            
            # Write frames
            for frame in frames:
                # Ensure frame is in correct format
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    out.write(frame)
                else:
                    self.logger.warning("Skipping frame with incorrect format")
            
            out.release()
            self.logger.info(f"Video saved successfully: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating video: {e}")
            return False
    
    def get_video_info(self, video_path: str) -> Optional[dict]:
        """
        Get information about a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information or None if error
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return None
            
            info = {
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            }
            
            cap.release()
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting video info: {e}")
            return None
    
    def resize_frame(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize frame to target size while maintaining aspect ratio.
        
        Args:
            frame: Input frame
            target_size: (width, height) target size
            
        Returns:
            Resized frame
        """
        target_width, target_height = target_size
        height, width = frame.shape[:2]
        
        # Calculate scaling factor to maintain aspect ratio
        scale = min(target_width / width, target_height / height)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize frame
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Add padding if necessary to reach target size
        if new_width != target_width or new_height != target_height:
            # Create black canvas
            canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            
            # Calculate padding
            y_offset = (target_height - new_height) // 2
            x_offset = (target_width - new_width) // 2
            
            # Place resized frame on canvas
            canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
            
            return canvas
        
        return resized
    
    def extract_frames_at_intervals(self, 
                                  video_path: str, 
                                  interval_seconds: float = 1.0) -> List[Tuple[np.ndarray, float]]:
        """
        Extract frames at specific time intervals.
        
        Args:
            video_path: Path to video file
            interval_seconds: Time interval between extracted frames
            
        Returns:
            List of (frame, timestamp) tuples
        """
        frames_with_timestamps = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                self.logger.error(f"Could not open video: {video_path}")
                return frames_with_timestamps
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps * interval_seconds)
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    timestamp = frame_idx / fps
                    frames_with_timestamps.append((frame.copy(), timestamp))
                
                frame_idx += 1
            
            cap.release()
            
        except Exception as e:
            self.logger.error(f"Error extracting frames at intervals: {e}")
        
        return frames_with_timestamps
    
    def preprocess_frame(self, frame: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Preprocess frame for better segmentation results.
        
        Args:
            frame: Input frame
            target_size: Optional target size (width, height)
            
        Returns:
            Preprocessed frame
        """
        processed = frame.copy()
        
        # Resize if target size specified
        if target_size:
            processed = self.resize_frame(processed, target_size)
        
        # Apply slight Gaussian blur to reduce noise
        processed = cv2.GaussianBlur(processed, (3, 3), 0)
        
        # Enhance contrast slightly
        processed = cv2.convertScaleAbs(processed, alpha=1.1, beta=10)
        
        return processed