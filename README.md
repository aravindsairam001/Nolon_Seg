# Floor/Pathway Segmentation for First-Person Videos

This project provides a comprehensive solution for segmenting floors and pathways from first-person perspective videos using the Segment Anything Model (SAM) and computer vision techniques.

## Features

- **Multiple Detection Methods**: Combines geometric, color-based, texture-based, and edge-based detection
- **SAM Integration**: Uses Segment Anything Model for precise segmentation
- **Fallback Mechanisms**: Works even without SAM using traditional CV methods
- **Comprehensive Visualization**: Creates overlays, statistics, and progress visualizations
- **Flexible Configuration**: Easy-to-modify settings for different scenarios

## Installation

1. **Clone or download this project**

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download SAM model (optional but recommended):**
   
   The script will automatically prompt you to download a SAM model. Choose one based on your needs:
   
   - **vit_b** (358MB): Fastest, good quality
   - **vit_l** (1.25GB): Balanced speed/quality
   - **vit_h** (2.56GB): Best quality, slowest
   
   Models will be downloaded to the `models/` directory.

## Usage

### Basic Usage

Place your video file in the `input_videos/` directory and run:

```bash
python floor_segmentation.py input_videos/your_video.mp4
```

### Advanced Usage

```bash
python floor_segmentation.py input_videos/your_video.mp4 \
  --output custom_output_dir \
  --sam-model models/sam_vit_b_01ec64.pth \
  --device cuda \
  --frame-skip 2 \
  --confidence 0.6
```

### Parameters

- `input_video`: Path to your input video file
- `--output`: Output directory (default: "output")
- `--sam-model`: Path to SAM model checkpoint
- `--device`: Device to use ("auto", "cpu", "cuda")
- `--frame-skip`: Process every nth frame (default: 1)
- `--confidence`: Confidence threshold for detection (default: 0.5)

## Project Structure

```
Nolon_Seg/
├── floor_segmentation.py      # Main script
├── requirements.txt           # Python dependencies
├── config.ini                # Configuration settings
├── README.md                 # This file
├── input_videos/             # Place your videos here
├── models/                   # SAM model checkpoints
├── output/                   # Generated outputs
│   ├── frames/              # Extracted frames
│   ├── masks/               # Segmentation masks
│   └── videos/              # Output videos
└── utils/                    # Utility modules
    ├── __init__.py
    ├── video_processor.py    # Video I/O operations
    ├── floor_detector.py     # Floor detection algorithms
    └── visualizer.py         # Visualization tools
```

## How It Works

### 1. Multi-Method Floor Detection

The system uses several complementary approaches:

- **Geometric Constraints**: Floors typically appear in the lower portion of first-person videos
- **Color Clustering**: Groups similar colors and identifies dominant floor colors
- **Texture Analysis**: Floors have characteristic texture patterns
- **Edge Detection**: Identifies horizontal lines that often mark floor boundaries

### 2. SAM Integration

When available, the Segment Anything Model provides precise segmentation:
- Geometric methods identify candidate floor regions
- These regions generate point prompts for SAM
- SAM produces detailed, accurate segmentation masks

### 3. Fallback Methods

If SAM is unavailable, the system uses traditional computer vision:
- Combines multiple detection methods
- Applies spatial filtering and morphological operations
- Ensures robust performance across different scenarios

## Output Files

After processing, you'll find:

- **frames/**: Individual video frames
- **masks/**: Binary masks showing detected floor regions
- **videos/**: Output video with floor overlay
- **summary.txt**: Processing statistics
- **statistics.png**: Visualization of detection results

## Configuration

Modify `config.ini` to adjust:
- Detection sensitivity
- Visualization colors
- Output quality
- Processing parameters

## Tips for Best Results

1. **Video Quality**: Higher resolution videos generally produce better results
2. **Lighting**: Consistent lighting helps with color-based detection
3. **Frame Rate**: For long videos, increase `--frame-skip` to speed up processing
4. **Model Choice**: Use `vit_h` for best quality, `vit_b` for speed
5. **Confidence Threshold**: Lower values detect more floor area but may include false positives

## Troubleshooting

### Common Issues

1. **SAM Installation Issues**:
   ```bash
   pip install git+https://github.com/facebookresearch/segment-anything.git
   ```

2. **CUDA Memory Issues**:
   - Use `--device cpu` for CPU-only processing
   - Reduce video resolution
   - Increase `--frame-skip`

3. **Poor Detection Results**:
   - Adjust `--confidence` threshold
   - Modify detection parameters in `config.ini`
   - Ensure good lighting in source video

### Performance Tips

- Use GPU acceleration with `--device cuda`
- Process every 2nd or 3rd frame with `--frame-skip`
- Use smaller SAM model (`vit_b`) for faster processing

## Requirements

- Python 3.7+
- OpenCV 4.5+
- PyTorch 1.9+
- 4GB+ RAM (8GB+ recommended)
- GPU with 4GB+ VRAM (optional but recommended)

## License

This project is provided as-is for educational and research purposes. SAM model usage follows Facebook's license terms.