# Floor/Pathway Segmentation and Trajectory Planning for First-Person Videos

This project provides a comprehensive solution for segmenting floors and pathways from first-person perspective videos using the Segment Anything Model (SAM) and computer vision techniques, with integrated trajectory planning and obstacle detection for robot navigation.

## Features

### Floor Segmentation
- **Multiple Detection Methods**: Combines geometric, color-based, texture-based, and edge-based detection
- **SAM Integration**: Uses Segment Anything Model for precise segmentation
- **Fallback Mechanisms**: Works even without SAM using traditional CV methods
- **Comprehensive Visualization**: Creates overlays, statistics, and progress visualizations

### Trajectory Planning
- **Obstacle Detection**: Multi-method obstacle detection (height-based, edge-based, contrast-based)
- **A* Path Planning**: Efficient pathfinding with obstacle avoidance
- **Robot Configuration**: Default 3m robot width with customizable parameters
- **Safety Analysis**: Collision risk assessment and emergency stop detection
- **Real-time Planning**: Frame-by-frame trajectory generation

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

### Trajectory Planning (Main System)

Place your video file in the `input_videos/` directory and run:

```bash
python trajectory_planning.py input_videos/your_video.mp4
```

### Advanced Trajectory Planning

```bash
python trajectory_planning.py input_videos/your_video.mp4 \
  --output custom_output_dir \
  --robot-width 2.5 \
  --sam-model models/sam_vit_b_01ec64.pth \
  --device cuda \
  --frame-skip 2 \
  --confidence 0.6
```

### Floor Segmentation Only

For basic floor segmentation without trajectory planning:

```bash
python floor_segmentation.py input_videos/your_video.mp4
```

### Parameters

- `input_video`: Path to your input video file
- `--output`: Output directory (default: "output")
- `--robot-width`: Robot width in meters (default: 3.0)
- `--sam-model`: Path to SAM model checkpoint
- `--device`: Device to use ("auto", "cpu", "cuda")
- `--frame-skip`: Process every nth frame (default: 1)
- `--confidence`: Confidence threshold for detection (default: 0.5)

## Project Structure

```
Nolon_Seg/
├── trajectory_planning.py     # Main trajectory planning script
├── floor_segmentation.py      # Basic floor segmentation script
├── requirements.txt           # Python dependencies
├── config.ini                # Configuration settings
├── README.md                 # This file
├── input_videos/             # Place your videos here
├── models/                   # SAM model checkpoints
├── output/                   # Generated outputs
│   ├── frames/              # Extracted frames
│   ├── masks/               # Floor segmentation masks
│   ├── obstacle_maps/       # Detected obstacles
│   ├── trajectories/        # Planned trajectories
│   ├── videos/              # Output videos with trajectories
│   └── analysis/            # Safety analysis reports
└── utils/                    # Utility modules
    ├── __init__.py
    ├── video_processor.py    # Video I/O operations
    ├── floor_detector.py     # Floor detection algorithms
    ├── visualizer.py         # Visualization tools
    ├── obstacle_detector.py  # Obstacle detection methods
    ├── path_planner.py       # A* pathfinding algorithm
    ├── robot_config.py       # Robot configuration (3m default)
    └── trajectory_pipeline.py # Integrated planning pipeline
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

### 3. Obstacle Detection

Multi-method obstacle detection identifies navigation hazards:
- **Height-based Detection**: Uses depth or stereo information
- **Edge-based Detection**: Identifies sharp edges and discontinuities
- **Contrast-based Detection**: Detects objects with different textures/colors

### 4. Trajectory Planning

A* algorithm plans safe robot paths:
- Uses detected floor regions as navigable space
- Inflates obstacles based on robot width (default: 3m)
- Generates waypoints with safety margins
- Provides collision risk assessment

### 5. Robot Configuration

Simplified robot setup with 3m default width:
- Configurable robot dimensions and safety parameters
- Automatic safety margin calculation
- Emergency stop distance computation

## Output Files

After processing, you'll find:

### Trajectory Planning Output
- **frames/**: Individual video frames
- **masks/**: Binary masks showing detected floor regions
- **obstacle_maps/**: Detected obstacles and hazards
- **trajectories/**: Planned robot paths as JSON data
- **videos/**: Output video with trajectory overlays
- **analysis/**: Safety analysis and statistics reports

### Basic Floor Segmentation Output
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