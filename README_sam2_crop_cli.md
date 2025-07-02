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
# Process a single image
python sam2_crop_cli.py input.jpg output_dir/

# Process a directory of images
python sam2_crop_cli.py input_dir/ output_dir/
```

### Advanced Usage

```bash
# Process with custom area thresholds
python sam2_crop_cli.py input_dir/ output_dir/ --min-area 1000 --max-area 50000

# Use different SAM2 model
python sam2_crop_cli.py input_dir/ output_dir/ --model sam2_hiera_l

# Custom output size
python sam2_crop_cli.py input_dir/ output_dir/ --output-size 512 512

# Add padding and remove holes
python sam2_crop_cli.py input_dir/ output_dir/ --padding 20 --hole-size 10

# Use transparent background instead of gray
python sam2_crop_cli.py input_dir/ output_dir/ --no-gray-bg

# Custom gray background value
python sam2_crop_cli.py input_dir/ output_dir/ --gray-value 200
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
- `--model`: SAM2 model type - `sam2_hiera_b+`, `sam2_hiera_l`, `sam2_hiera_s`, or `sam2_hiera_t` (default: sam2_hiera_b+)
- `--device`: Device to run inference on - `cuda` or `cpu` (default: cpu)

#### Processing Settings
- `--padding`: Padding size in pixels around segments (default: 10)
- `--hole-size`: Size of holes to remove using morphological operations (default: 5)
- `--output-size`: Output image size as WIDTH HEIGHT (default: 1024 1024)

#### Background Settings
- `--gray-bg`: Use gray background (default: True)
- `--no-gray-bg`: Use transparent (black) background
- `--gray-value`: Gray background value 0-255 (default: 128)

#### Other
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
└── ...
```

Each image gets its own subdirectory containing the cropped segments.

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
    --model sam2_hiera_l \
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

## Tips

1. **Area Thresholds**: Start with wide ranges and narrow down based on your needs
2. **Model Selection**: 
   - `sam2_hiera_b+`: Fastest, good for most use cases
   - `sam2_hiera_l`: Better quality, slower
   - `sam2_hiera_s`: Best quality, slowest
3. **Padding**: Use larger padding if you need to include more context around objects
4. **Hole Removal**: Use larger hole_size values to remove bigger holes in segments
5. **Output Size**: Larger sizes preserve more detail but use more storage

## Troubleshooting

### Common Issues

1. **No segments found**: Try reducing `--min-area` or increasing `--max-area`
2. **Too many small segments**: Increase `--min-area`
3. **Missing boundary pixels**: Increase `--padding`
4. **Holes in segments**: Increase `--hole-size`
5. **Out of memory**: Use smaller model or reduce `--output-size`

### Performance Tips

- Use GPU (`--device cuda`) for faster processing
- Use smaller models for faster processing
- Process images in smaller batches if memory is limited
- Use smaller output sizes to save storage space

## License

This tool is part of the SAM2 project and follows the same license terms. 