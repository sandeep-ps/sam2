# SAM2 Image Cropping CLI Tool

A command-line interface tool that uses SAM2 (Segment Anything Model 2) to automatically segment images and crop individual objects based on area thresholds.

## Features

- **Automatic Segmentation**: Uses SAM2's automatic mask generator to find objects in images
- **Area-based Filtering**: Filter segments by minimum and maximum area thresholds
- **Batch Processing**: Process entire directories of images
- **Configurable Output**: Customize output size, padding, and background
- **Background Removal**: Automatically removes the largest segment (typically background)
- **Hole Removal**: Remove small holes in segments using morphological operations
- **Padding**: Add padding around segments to include boundary pixels
- **Custom Background Colors**: Support for gray, transparent, or custom RGB backgrounds
- **Segment Sorting**: Sort segments by Y coordinate (top-to-bottom or bottom-to-top)
- **Debug Mode**: Save intermediate images for debugging
- **Robust Error Handling**: Automatic CUDA fallback and memory management

## Installation

1. Make sure you have SAM2 installed:
```bash
pip install -e .
```

2. Install additional dependencies:
```bash
pip install opencv-python pycocotools
```

3. Make the script executable:
```bash
chmod +x sam2_crop_cli.py
```

## Usage

### Basic Usage

```bash
# Process a single image (uses default sam2.1_hiera_b+ model)
python sam2_crop_cli.py input.jpg output_dir/

# Process a directory of images (uses default sam2.1_hiera_b+ model)
python sam2_crop_cli.py input_dir/ output_dir/
```

**Note**: By default, the tool uses the `sam2.1_hiera_b+` (SAM2.1 base plus) model with config file `configs/sam2.1/sam2.1_hiera_b+.yaml` and checkpoint `checkpoints/sam2.1_hiera_base_plus.pt`.

### Advanced Usage

```bash
# Process with custom area thresholds
python sam2_crop_cli.py input_dir/ output_dir/ --min-area 1000 --max-area 50000

# Use different SAM2 model by specifying config and checkpoint
python sam2_crop_cli.py input_dir/ output_dir/ --config-file configs/sam2.1/sam2.1_hiera_l.yaml --ckpt-path checkpoints/sam2.1_hiera_large.pt

# Use original SAM2 models (for compatibility)
python sam2_crop_cli.py input_dir/ output_dir/ --config-file configs/sam2/sam2_hiera_b+.yaml --ckpt-path checkpoints/sam2_hiera_base_plus.pt

# Custom output size
python sam2_crop_cli.py input_dir/ output_dir/ --output-size 512 512

# Add padding and remove holes
python sam2_crop_cli.py input_dir/ output_dir/ --padding 20 --hole-size 10

# Use transparent background instead of gray
python sam2_crop_cli.py input_dir/ output_dir/ --no-gray-bg

# Use custom background color (red)
python sam2_crop_cli.py input_dir/ output_dir/ --bg-color 255 0 0

# Custom gray background value
python sam2_crop_cli.py input_dir/ output_dir/ --gray-value 200

# Sort segments by Y coordinate in descending order (top to bottom)
python sam2_crop_cli.py input_dir/ output_dir/ --sort-by-y descending

# Sort segments by Y coordinate in ascending order (bottom to top, default)
python sam2_crop_cli.py input_dir/ output_dir/ --sort-by-y ascending

# Save debug images for troubleshooting
python sam2_crop_cli.py input_dir/ output_dir/ --save-debug
```

## Command Line Arguments

### Required Arguments
- `input`: Input image file or directory
- `output`: Output directory

### Optional Arguments

#### Area Thresholds
- `--min-area`: Minimum area threshold in pixels (default: 100)
- `--max-area`: Maximum area threshold in pixels (default: 1000000)

#### Model Settings
- `--device`: Device to run inference on - `cuda` or `cpu` (default: cpu)
- `--config-file`: Path to model config file (default: configs/sam2.1/sam2.1_hiera_b+.yaml)
- `--ckpt-path`: Path to model checkpoint file (default: checkpoints/sam2.1_hiera_base_plus.pt)

#### Processing Settings
- `--padding`: Padding size in pixels around segments (default: 10)
- `--hole-size`: Size of holes to remove using morphological operations (default: 5)
- `--output-size`: Output image size as WIDTH HEIGHT (default: 1024 1024)

#### Background Settings
- `--gray-bg`: Use gray background (default: True when no background option specified)
- `--no-gray-bg`: Use transparent (black) background
- `--bg-color`: Custom background color as RGB values (0-255 each, e.g., 255 0 0 for red)
- `--gray-value`: Gray background value 0-255 (default: 128)

#### Sorting Settings
- `--sort-by-y`: Sort segments by Y coordinate - `ascending` (bottom to top) or `descending` (top to bottom) (default: ascending)

#### Other
- `--save-debug`: Save resized/original images for debugging
- `--verbose`, `-v`: Enable verbose logging

## Output Structure

The tool creates the following directory structure:

```
output_dir/
├── image1/
│   ├── image1_segment_000.png
│   ├── image1_segment_001.png
│   └── ...
├── image2/
│   ├── image2_segment_000.png
│   ├── image2_segment_001.png
│   └── ...
└── debug/  # Only if --save-debug is used
    ├── image1_resized.png
    ├── image1_original.png
    └── ...
```

Each image gets its own subdirectory containing the cropped segments. When using `--save-debug`, a `debug` subdirectory is also created with intermediate images.

## Examples

### Example 1: Basic Processing
```bash
# Process all images in a directory with default settings
python sam2_crop_cli.py ./images/ ./output/
```

### Example 2: Filter Large Objects
```bash
# Only keep segments between 5000 and 50000 pixels
python sam2_crop_cli.py ./images/ ./output/ --min-area 5000 --max-area 50000
```

### Example 3: High-Quality Output
```bash
# Use large model, high resolution, and extra padding
python sam2_crop_cli.py ./images/ ./output/ \
    --config-file configs/sam2.1/sam2.1_hiera_l.yaml \
    --ckpt-path checkpoints/sam2.1_hiera_large.pt \
    --output-size 2048 2048 \
    --padding 30 \
    --hole-size 10
```

### Example 4: Small Objects
```bash
# Focus on small objects with minimal padding
python sam2_crop_cli.py ./images/ ./output/ \
    --min-area 100 \
    --max-area 5000 \
    --padding 5 \
    --output-size 512 512
```

### Example 5: Custom Background
```bash
# Use red background for all segments
python sam2_crop_cli.py ./images/ ./output/ \
    --bg-color 255 0 0 \
    --padding 15
```

### Example 6: Sorted Output
```bash
# Sort segments from top to bottom (useful for reading order)
python sam2_crop_cli.py ./images/ ./output/ \
    --sort-by-y descending \
    --min-area 1000 \
    --max-area 50000
```

### Example 7: Debug Mode
```bash
# Process with debug images to troubleshoot issues
python sam2_crop_cli.py ./images/ ./output/ \
    --save-debug \
    --verbose
```

## Available Models

The tool supports both SAM2.1 and original SAM2 models. You can specify any model by providing the appropriate config file and checkpoint:

### SAM2.1 Models (Recommended)
- **Base Plus**: `configs/sam2.1/sam2.1_hiera_b+.yaml` + `checkpoints/sam2.1_hiera_base_plus.pt` (default)
- **Large**: `configs/sam2.1/sam2.1_hiera_l.yaml` + `checkpoints/sam2.1_hiera_large.pt`
- **Small**: `configs/sam2.1/sam2.1_hiera_s.yaml` + `checkpoints/sam2.1_hiera_small.pt`
- **Tiny**: `configs/sam2.1/sam2.1_hiera_t.yaml` + `checkpoints/sam2.1_hiera_tiny.pt`

### Original SAM2 Models
- **Base Plus**: `configs/sam2/sam2_hiera_b+.yaml` + `checkpoints/sam2_hiera_base_plus.pt`
- **Large**: `configs/sam2/sam2_hiera_l.yaml` + `checkpoints/sam2_hiera_large.pt`
- **Small**: `configs/sam2/sam2_hiera_s.yaml` + `checkpoints/sam2_hiera_small.pt`
- **Tiny**: `configs/sam2/sam2_hiera_t.yaml` + `checkpoints/sam2_hiera_tiny.pt`

## Tips

1. **Area Thresholds**: Start with wide ranges and narrow down based on your needs
2. **Model Selection**: 
   - **SAM2.1 Models** (default): Latest version with improved performance and better segmentation quality
   - **SAM2 Models**: Original models available for compatibility
3. **Padding**: Use larger padding if you need to include more context around objects
4. **Hole Removal**: Use larger hole_size values to remove bigger holes in segments
5. **Output Size**: Larger sizes preserve more detail but use more storage space
6. **Background Options**: 
   - Default gray background provides good contrast
   - Transparent background (--no-gray-bg) is useful for compositing
   - Custom colors (--bg-color) can match your workflow needs
7. **Segment Sorting**: 
   - Use `--sort-by-y descending` for reading order (top to bottom)
   - Use `--sort-by-y ascending` for reverse reading order (bottom to top)

## Troubleshooting

### Common Issues

1. **No segments found**: Try reducing `--min-area` or increasing `--max-area`
2. **Too many small segments**: Increase `--min-area`
3. **Missing boundary pixels**: Increase `--padding`
4. **Holes in segments**: Increase `--hole-size`
5. **Out of memory**: Use smaller model or reduce `--output-size`
6. **CUDA errors**: The tool automatically falls back to CPU if CUDA issues occur

### Performance Tips

- Use GPU (`--device cuda`) for faster processing
- Use smaller models for faster processing
- Process images in smaller batches if memory is limited
- Use smaller output sizes to save storage space

### Debug Mode

Use `--save-debug` to save intermediate images (resized input and original) alongside the segments. This helps troubleshoot issues with:
- Input image processing
- Segmentation quality
- Area filtering
- Background handling

The debug images are saved in a `debug/` subdirectory within your output directory.

### Memory Management

The tool includes several memory management features:
- Automatic CUDA memory clearing between images
- Automatic fallback to CPU if GPU memory is insufficient
- Image resizing for large images to prevent memory issues
- Garbage collection to free memory

## License

This tool is part of the SAM2 project and follows the same license terms. 