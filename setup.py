#!/usr/bin/env python3
"""
Setup script for Floor Segmentation Pipeline
Downloads SAM models and sets up the environment.
"""

import os
import urllib.request
import argparse
from pathlib import Path

def download_sam_model(model_type: str = "vit_b") -> bool:
    """Download SAM model checkpoint."""
    
    model_urls = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", 
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    }
    
    model_sizes = {
        "vit_b": "358MB",
        "vit_l": "1.25GB",
        "vit_h": "2.56GB"
    }
    
    if model_type not in model_urls:
        print(f"Invalid model type: {model_type}")
        print(f"Available types: {list(model_urls.keys())}")
        return False
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    url = model_urls[model_type]
    filename = url.split("/")[-1]
    filepath = models_dir / filename
    
    if filepath.exists():
        print(f"Model {model_type} already exists at {filepath}")
        return True
    
    print(f"Downloading SAM model: {model_type} ({model_sizes[model_type]})")
    print(f"URL: {url}")
    print(f"Destination: {filepath}")
    
    try:
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) // total_size)
                print(f"\rProgress: {percent}% ({downloaded // 1024 // 1024}MB)", end="")
        
        urllib.request.urlretrieve(url, filepath, progress_hook)
        print(f"\nDownload completed: {filepath}")
        return True
        
    except Exception as e:
        print(f"\nDownload failed: {e}")
        if filepath.exists():
            filepath.unlink()  # Remove partial download
        return False

def main():
    parser = argparse.ArgumentParser(description="Setup Floor Segmentation Pipeline")
    parser.add_argument("--model", default="vit_b", choices=["vit_b", "vit_l", "vit_h"],
                       help="SAM model to download (default: vit_b)")
    parser.add_argument("--skip-model", action="store_true", 
                       help="Skip model download")
    
    args = parser.parse_args()
    
    print("=== Floor Segmentation Pipeline Setup ===")
    
    # Create directories
    directories = ["input_videos", "output", "output/frames", "output/masks", "output/videos", "models"]
    for dir_name in directories:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_name}")
    
    # Download SAM model
    if not args.skip_model:
        print(f"\nDownloading SAM model: {args.model}")
        success = download_sam_model(args.model)
        if success:
            print("✓ SAM model downloaded successfully")
        else:
            print("✗ SAM model download failed")
            print("You can run the pipeline without SAM (reduced accuracy)")
    else:
        print("⚠ Skipping SAM model download")
    
    print("\n=== Setup Complete ===")
    print("\nNext steps:")
    print("1. Place your video files in the 'input_videos/' directory")
    print("2. Run: python floor_segmentation.py input_videos/your_video.mp4")
    print("3. Check the 'output/' directory for results")
    
    print(f"\nFor help: python floor_segmentation.py --help")

if __name__ == "__main__":
    main()