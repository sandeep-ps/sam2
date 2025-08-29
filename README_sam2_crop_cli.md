# SAM2 Image Cropping CLI Tool

A command-line interface tool that uses SAM2 (Segment Anything Model 2) to automatically segment images and crop individual objects based on area thresholds.

## Features

- **Automatic Segmentation**: Uses SAM2's automatic mask generator to find objects in images
- **Area-based Filtering**: Filter segments by minimum and maximum area thresholds
- **Batch Processing**: Process entire directories of images
- **Configurable Output**: Customize output size, padding, and background
- **Background Removal**: Automatically removes the largest segment (typically background)
- **Enhanced Segmentation**: Improved SAM2 parameters for better accuracy and coverage
- **Multi-Scale Processing**: Process images at multiple scales for comprehensive segmentation
- **Advanced Post-Processing**: Multi-step mask cleaning with morphological operations
- **Hole and Island Removal**: Remove small holes and disconnected regions using SAM2 utilities
- **Padding**: Add padding around segments to include boundary pixels
- **Custom Background Colors**: Support for gray, transparent, or custom RGB backgrounds
- **Segment Sorting**: Sort segments by Y coordinate (top-to-bottom or bottom-to-top)
- **Mask Resizing**: Option to resize masks to original image dimensions for higher quality segments
- **Debug Mode**: Save intermediate images for debugging
- **Overwrite Control**: Option to overwrite existing segment images or skip them
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

# Enhanced processing with improved parameters
python sam2_crop_cli.py input_dir/ output_dir/ --padding 15 --hole-size 8 --morph-kernel-size 5

# Multi-scale processing for maximum coverage
python sam2_crop_cli.py input_dir/ output_dir/ --multi-scale --scales 0.75 1.0 1.25

# Use transparent background instead of gray
python sam2_crop_cli.py input_dir/ output_dir/ --no-gray-bg

# Use custom background color (red)
python sam2_crop_cli.py input_dir/ output_dir/ --bg-color 255 0 0

# Custom gray background value
python sam2_crop_cli.py input_dir/ output_dir/ --gray-value 200

# Use enhanced minimum mask region area
python sam2_crop_cli.py input_dir/ output_dir/ --min-mask-region-area 256

# Use enhanced maximum resize dimension
python sam2_crop_cli.py input_dir/ output_dir/ --max-resize-dimension 2048

# Sort segments by Y coordinate in descending order (top to bottom)
python sam2_crop_cli.py input_dir/ output_dir/ --sort-by-y descending

# Sort segments by Y coordinate in ascending order (bottom to top, default)
python sam2_crop_cli.py input_dir/ output_dir/ --sort-by-y ascending

# Resize masks to original image dimensions before cropping
python sam2_crop_cli.py input_dir/ output_dir/ --resize-mask-to-original

# Save debug images for troubleshooting
python sam2_crop_cli.py input_dir/ output_dir/ --save-debug

# Overwrite existing segment images
python sam2_crop_cli.py input_dir/ output_dir/ --overwrite

# Use multi-scale processing for better segmentation coverage
python sam2_crop_cli.py input_dir/ output_dir/ --multi-scale --scales 0.75 1.0 1.25
```

### Usage Full Details

```
usage: sam2_crop_cli.py [-h] [--min-area MIN_AREA] [--max-area MAX_AREA] [--device {cuda,cpu}] [--config-file CONFIG_FILE]
                        [--ckpt-path CKPT_PATH] [--min-mask-region-area MIN_MASK_REGION_AREA]
                        [--max-resize-dimension MAX_RESIZE_DIMENSION] [--padding PADDING] [--hole-size HOLE_SIZE]
                        [--morph-kernel-size MORPH_KERNEL_SIZE] [--output-size WIDTH HEIGHT] [--gray-bg] [--no-gray-bg]
                        [--bg-color R G B] [--gray-value GRAY_VALUE] [--sort-by-y {ascending,descending}]
                        [--resize-mask-to-original] [--multi-scale] [--scales SCALES [SCALES ...]] [--overwrite]
                        [--save-debug] [--verbose]
                        input output

SAM2 Image Cropping Tool - Automatically crop image segments using SAM2

positional arguments:
  input                 Input image file or directory
  output                Output directory

options:
  -h, --help            show this help message and exit
  --min-area MIN_AREA   Minimum area threshold in pixels (default: 100)
  --max-area MAX_AREA   Maximum area threshold in pixels (default: 1000000)
  --device {cuda,cpu}   Device to run inference on (default: cpu)
  --config-file CONFIG_FILE
                        Path to model config file (default: configs/sam2.1/sam2.1_hiera_b+.yaml)
  --ckpt-path CKPT_PATH
                        Path to model checkpoint file (default: checkpoints/sam2.1_hiera_base_plus.pt)
  --min-mask-region-area MIN_MASK_REGION_AREA
                        Minimum mask region area in pixels (default: 256, reduced for better detail)
  --max-resize-dimension MAX_RESIZE_DIMENSION
                        Maximum dimension for image resizing (default: 2048, increased for better detail)
  --padding PADDING     Padding size in pixels (default: 15, increased for better coverage)
  --hole-size HOLE_SIZE
                        Size of holes to remove (default: 8, increased for better cleaning)
  --morph-kernel-size MORPH_KERNEL_SIZE
                        Kernel size for morphological operations (default: 5)
  --output-size WIDTH HEIGHT
                        Output image size (default: 1024 1024)
  --gray-bg             Use gray background (default: True)
  --no-gray-bg          Use transparent (black) background
  --bg-color R G B      Custom background color as RGB values (0-255 each)
  --gray-value GRAY_VALUE
                        Gray background value (0-255, default: 128)
  --sort-by-y {ascending,descending}
                        Sort segments by Y coordinate (default: ascending)
  --resize-mask-to-original
                        Resize masks to original image dimensions before cropping (useful when input was resized)
  --multi-scale         Use multi-scale processing for better segmentation coverage
  --scales SCALES [SCALES ...]
                        Scales for multi-scale processing (default: 0.75 1.0 1.25)
  --overwrite           Overwrite existing segment images (default: skip existing segments)
  --save-debug          Save resized/original images for debugging
  --verbose, -v         Enable verbose logging
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
- `--min-mask-region-area`: Minimum mask region area in pixels (default: 256, reduced for better detail)
- `--max-resize-dimension`: Maximum dimension for image resizing (default: 2048, increased for better detail)

#### Processing Settings
- `--padding`: Padding size in pixels around segments (default: 15, increased for better coverage)
- `--hole-size`: Size of holes to remove using morphological operations (default: 8, increased for better cleaning)
- `--morph-kernel-size`: Kernel size for morphological operations (default: 5)
- `--output-size`: Output image size as WIDTH HEIGHT (default: 1024 1024)

#### Background Settings
- `--gray-bg`: Use gray background (default: True when no background option specified)
- `--no-gray-bg`: Use transparent (black) background
- `--bg-color`: Custom background color as RGB values (0-255 each, e.g., 255 0 0 for red)
- `--gray-value`: Gray background value 0-255 (default: 128)

#### Sorting Settings
- `--sort-by-y`: Sort segments by Y coordinate - `ascending` (bottom to top) or `descending` (top to bottom) (default: ascending)

#### Mask Processing Settings
- `--resize-mask-to-original`: Resize masks to original image dimensions before cropping (useful when input was resized)
- `--multi-scale`: Use multi-scale processing for better segmentation coverage
- `--scales`: Custom scales for multi-scale processing (default: 0.75 1.0 1.25)

#### Other
- `--overwrite`: Overwrite existing segment images (default: skip existing segments)
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

### Example 3: High-Quality Output with Enhanced Processing
```bash
# Use large model, high resolution, and enhanced processing parameters
python sam2_crop_cli.py ./images/ ./output/ \
    --config-file configs/sam2.1/sam2.1_hiera_l.yaml \
    --ckpt-path checkpoints/sam2.1_hiera_large.pt \
    --output-size 2048 2048 \
    --padding 15 \
    --hole-size 8 \
    --morph-kernel-size 5 \
    --multi-scale \
    --scales 0.75 1.0 1.25
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

### Example 8: High-Quality Segments with Original Dimensions
```bash
# Resize masks to original image dimensions for higher quality segments
python sam2_crop_cli.py ./images/ ./output/ \
    --resize-mask-to-original \
    --min-area 1000 \
    --max-area 50000 \
    --padding 15
```

### Example 9: Enhanced Minimum Mask Region Area
```bash
# Use enhanced minimum mask region area for better detail preservation
python sam2_crop_cli.py ./images/ ./output/ \
    --min-mask-region-area 256 \
    --min-area 1000 \
    --max-area 50000
```

### Example 10: Enhanced Maximum Resize Dimension
```bash
# Use enhanced maximum resize dimension for higher quality processing of large images
python sam2_crop_cli.py ./images/ ./output/ \
    --max-resize-dimension 2048 \
    --min-area 1000 \
    --max-area 50000
```

### Example 11: Multi-Scale Processing
```bash
# Use multi-scale processing for maximum segmentation coverage
python sam2_crop_cli.py ./images/ ./output/ \
    --multi-scale \
    --scales 0.75 1.0 1.25 \
    --min-area 1000 \
    --max-area 50000
```

### Example 12: Overwrite Existing Segments
```bash
# Overwrite existing segment images instead of skipping them
python sam2_crop_cli.py ./images/ ./output/ \
    --overwrite \
    --min-area 1000 \
    --max-area 50000
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
3. **Enhanced Processing**: The tool now uses improved SAM2 parameters by default for better accuracy
4. **Multi-Scale Processing**: Use `--multi-scale` for comprehensive segmentation coverage across different scales
5. **Advanced Post-Processing**: The enhanced pipeline includes morphological operations for cleaner masks
6. **Padding**: Use larger padding if you need to include more context around objects (default increased to 15)
7. **Hole Removal**: Use larger hole_size values to remove bigger holes in segments (default increased to 8)
8. **Morphological Operations**: Use `--morph-kernel-size` to control the strength of mask cleaning operations
9. **Output Size**: Larger sizes preserve more detail but use more storage space
10. **Background Options**: 
    - Default gray background provides good contrast
    - Transparent background (--no-gray-bg) is useful for compositing
    - Custom colors (--bg-color) can match your workflow needs
11. **Segment Sorting**: 
    - Use `--sort-by-y descending` for reading order (top to bottom)
    - Use `--sort-by-y ascending` for reverse reading order (bottom to top)
12. **Mask Resizing**: 
    - Use `--resize-mask-to-original` when working with high-resolution images that get resized for processing
    - This ensures segments maintain the original image's detail and precision
    - Particularly useful when the input image is automatically resized (e.g., from 2048x2048 to 1024x1024) for memory management
13. **Enhanced Minimum Mask Region Area**: 
    - Default reduced to 256 pixels for better detail preservation
    - Higher values (e.g., 512) filter out small segments during generation, improving performance
    - Lower values (e.g., 128) allow smaller segments for more detailed segmentation
    - This parameter affects the initial segmentation, while `--min-area` filters the final results

14. **Enhanced Maximum Resize Dimension**: 
    - Default increased to 2048 pixels for better detail preservation
    - Higher values (e.g., 4096) preserve more detail but use more memory
    - Lower values (e.g., 1024) use less memory but may lose some detail
    - The enhanced default provides better quality while maintaining reasonable memory usage

15. **Multi-Scale Processing**: 
    - Use `--multi-scale` to process images at multiple scales (0.75x, 1.0x, 1.25x by default)
    - This ensures better coverage of objects at different sizes
    - Customize scales with `--scales` for specific use cases
    - Particularly effective for images with objects of varying sizes

16. **Overwrite Control**: 
    - By default, the tool skips existing segment images to avoid overwriting previous work
    - Use `--overwrite` to force regeneration of all segments, overwriting existing files
    - Useful when you want to update segments with new parameters or model settings
    - The tool logs which segments are being skipped when overwrite is disabled

## Troubleshooting

### Common Issues

1. **No segments found**: Try reducing `--min-area` or increasing `--max-area`
2. **Too many small segments**: Increase `--min-area` or `--min-mask-region-area`
3. **Missing boundary pixels**: Increase `--padding` (default is now 15)
4. **Holes in segments**: Increase `--hole-size` (default is now 8) or use `--morph-kernel-size`
5. **Poor segmentation quality**: Try `--multi-scale` for better coverage
6. **Out of memory**: Use smaller model or reduce `--output-size`
7. **CUDA errors**: The tool automatically falls back to CPU if CUDA issues occur
8. **Too many tiny segments during generation**: Increase `--min-mask-region-area` to filter out small segments earlier in the process
9. **Existing segments not being updated**: Use `--overwrite` to force regeneration of all segments
10. **OpenCV morphology errors**: The tool now properly handles boolean masks from SAM2 utilities

### Performance Tips

- Use GPU (`--device cuda`) for faster processing
- Use smaller models for faster processing
- Process images in smaller batches if memory is limited
- Use smaller output sizes to save storage space
- Use smaller `--max-resize-dimension` values to reduce memory usage
- Use larger `--max-resize-dimension` values for higher quality processing of large images
- **Multi-scale processing** increases processing time but provides better coverage
- **Enhanced default parameters** provide better quality but may use more memory
- **Morphological operations** add minimal overhead but significantly improve mask quality

### Debug Mode

Use `--save-debug` to save intermediate images (resized input and original) alongside the segments. This helps troubleshoot issues with:
- Input image processing
- Segmentation quality
- Area filtering
- Background handling
- Mask resizing behavior

The debug images are saved in a `debug/` subdirectory within your output directory.

### Memory Management

The tool includes several memory management features:
- Automatic CUDA memory clearing between images
- Automatic fallback to CPU if GPU memory is insufficient
- Image resizing for large images to prevent memory issues (controlled by `--max-resize-dimension`)
- Garbage collection to free memory

## License

This tool is part of the SAM2 project and follows the same license terms. 