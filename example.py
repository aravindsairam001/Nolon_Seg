#!/usr/bin/env python3
"""
Example usage of the Floor Segmentation Pipeline.
This script demonstrates how to use the pipeline programmatically.
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from floor_segmentation import FloorSegmentationPipeline

def run_example():
    """Run an example segmentation."""
    
    # Check for input video
    input_dir = Path("input_videos")
    video_files = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.avi")) + list(input_dir.glob("*.mov"))
    
    if not video_files:
        print("No video files found in input_videos/ directory.")
        print("Please add a video file and try again.")
        return
    
    # Use the first video file found
    video_path = str(video_files[0])
    print(f"Processing video: {video_path}")
    
    # Initialize pipeline
    print("Initializing floor segmentation pipeline...")
    pipeline = FloorSegmentationPipeline(
        sam_model_path=None,  # Auto-detect SAM model
        device="auto"         # Use GPU if available, otherwise CPU
    )
    
    # Process video
    output_dir = "output/example"
    print(f"Processing video with output to: {output_dir}")
    
    pipeline.process_video(
        video_path=video_path,
        output_dir=output_dir,
        frame_skip=2,          # Process every 2nd frame for speed
        confidence_threshold=0.5
    )
    
    print("\n=== Processing Complete ===")
    print(f"Check the '{output_dir}' directory for results:")
    print("- frames/: Individual video frames")
    print("- masks/: Floor segmentation masks")
    print("- videos/: Output video with floor overlay")
    print("- summary.txt: Processing statistics")

if __name__ == "__main__":
    print("=== Floor Segmentation Example ===")
    print("This example will process the first video found in input_videos/")
    
    # Check if setup is complete
    if not Path("input_videos").exists():
        print("Please run setup.py first to initialize the project structure.")
        sys.exit(1)
    
    run_example()