#!/usr/bin/env python3
"""
SAM2 Image Cropping CLI Tool

This tool uses SAM2 (Segment Anything Model 2) to automatically segment images
and crop individual objects based on area thresholds.
"""

import argparse
import os
import sys
import logging

# Set up logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check and import required dependencies
try:
    import cv2
except ImportError:
    logger.error("OpenCV (cv2) is not installed. Please install it with: pip install opencv-python")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    logger.error("NumPy is not installed. Please install it with: pip install numpy")
    sys.exit(1)

try:
    import torch
except ImportError:
    logger.error("PyTorch is not installed. Please install it with: pip install torch")
    sys.exit(1)

try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.utils.amg import remove_small_regions
except ImportError as e:
    logger.error(f"SAM2 is not properly installed: {e}")
    logger.error("Please install SAM2 with: pip install -e .")
    sys.exit(1)

# Check for optional dependencies
try:
    import pycocotools
except ImportError:
    logger.warning("pycocotools is not installed. Some mask formats may not work properly.")
    logger.warning("Install with: pip install pycocotools")

from pathlib import Path
from typing import List, Dict, Any, Tuple



class SAM2Cropper:
    """SAM2-based image cropper with configurable area thresholds."""
    
    def _setup_cuda_environment(self):
        """Setup CUDA environment with error handling."""
        try:
            # Set CUDA environment variables for debugging
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            
            # Check CUDA availability
            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                return True
            else:
                logger.info("CUDA not available, using CPU")
                return False
        except Exception as e:
            logger.warning(f"CUDA setup failed: {e}")
            return False
    
    def _safe_device_selection(self, requested_device: str) -> str:
        """Safely select device with fallback to CPU."""
        try:
            if requested_device == "cuda" and torch.cuda.is_available():
                # Check if CUDA memory is sufficient
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                if gpu_memory < 4.0:  # Less than 4GB
                    logger.warning(f"GPU memory ({gpu_memory:.1f}GB) may be insufficient, using CPU")
                    return "cpu"
                return "cuda"
            else:
                if requested_device == "cuda":
                    logger.warning("CUDA requested but not available, using CPU")
                return "cpu"
        except Exception as e:
            logger.warning(f"Device selection failed: {e}, using CPU")
            return "cpu"
    
    def __init__(self, device: str = "cpu", 
                 config_file: str = "configs/sam2.1/sam2.1_hiera_b+.yaml", ckpt_path: str = "checkpoints/sam2.1_hiera_base_plus.pt"):
        """
        Initialize SAM2 cropper.
        
        Args:
            device: Device to run inference on ('cuda' or 'cpu')
            config_file: Path to model config file (default: configs/sam2.1/sam2.1_hiera_b+.yaml)
            ckpt_path: Path to model checkpoint file (default: checkpoints/sam2.1_hiera_base_plus.pt)
        """
        # Setup CUDA environment
        self._setup_cuda_environment()
        
        # Safely select device
        self.device = self._safe_device_selection(device)
        logger.info(f"Using device: {self.device}")
        
        # Store paths for fallback
        self.config_file = config_file
        self.ckpt_path = ckpt_path
        
        # Build SAM2 model with error handling
        logger.info(f"Loading SAM2 model with config: {config_file}")
        try:
            logger.info(f"Using config: {config_file} and checkpoint: {ckpt_path}")
            self.model = build_sam2(
                config_file=config_file, 
                ckpt_path=ckpt_path, 
                device=self.device, 
                apply_postprocessing=False
            )
        except Exception as e:
            if "CUDA" in str(e) or "cuda" in str(e).lower() or "memory" in str(e).lower():
                logger.warning(f"CUDA/memory error detected, falling back to CPU: {e}")
                self.device = "cpu"
                self.model = build_sam2(
                    config_file=config_file, 
                    ckpt_path=ckpt_path, 
                    device="cpu", 
                    apply_postprocessing=False
                )
            else:
                raise e
        
        # Initialize automatic mask generator
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_side=64,
            points_per_batch=128,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            crop_overlap_ratio=0.5,
            min_mask_region_area=100,
        )
        
        logger.info(f"SAM2 model loaded successfully on device: {self.device}")
        
    def filter_masks_by_area(self, masks: List[Dict], min_area: int, max_area: int) -> List[Dict]:
        """
        Filter masks based on area thresholds.
        
        Args:
            masks: List of mask dictionaries
            min_area: Minimum area in pixels
            max_area: Maximum area in pixels
            
        Returns:
            Filtered list of masks
        """
        filtered_masks = []
        
        for mask in masks:
            area = mask['area']
            if min_area <= area <= max_area:
                filtered_masks.append(mask)
                
        logger.info(f"Filtered {len(masks)} masks to {len(filtered_masks)} based on area [{min_area}, {max_area}]")
        return filtered_masks
    
    def remove_background_mask(self, masks: List[Dict]) -> List[Dict]:
        """
        Remove the largest mask (typically background).
        
        Args:
            masks: List of mask dictionaries
            
        Returns:
            List of masks without the background
        """
        if not masks:
            return masks
            
        # Sort by area in descending order
        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        # Remove the largest mask (background)
        filtered_masks = sorted_masks[1:]
        
        logger.info(f"Removed background mask (area: {sorted_masks[0]['area']})")
        return filtered_masks
    
    def process_mask(self, mask: Dict, padding: int = 10, hole_size: int = 5) -> np.ndarray:
        """
        Process a single mask with padding and hole removal.
        
        Args:
            mask: Mask dictionary
            padding: Padding size in pixels
            hole_size: Size of holes to remove
            
        Returns:
            Processed binary mask
        """
        try:
            # Get binary mask
            if isinstance(mask['segmentation'], dict):
                try:
                    from pycocotools import mask as mask_utils
                    binary_mask = mask_utils.decode(mask['segmentation'])
                except ImportError:
                    logger.error("pycocotools is not installed. Please install it with: pip install pycocotools")
                    raise
            else:
                binary_mask = mask['segmentation'].astype(np.uint8)
            
            # Ensure mask is 2D
            if len(binary_mask.shape) > 2:
                binary_mask = binary_mask.squeeze()
            
            # Ensure mask is the right type
            if binary_mask.dtype != np.uint8:
                binary_mask = binary_mask.astype(np.uint8)
                
        except Exception as e:
            logger.error(f"Error processing mask: {e}")
            raise
        
        # Remove small holes
        if hole_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hole_size, hole_size))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # Add padding using dilation
        if padding > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (padding*2+1, padding*2+1))
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        
        return binary_mask
    
    def crop_and_center(self, image: np.ndarray, mask: np.ndarray, 
                       output_size: Tuple[int, int] = (1024, 1024),
                       gray_bg: bool = True, gray_value: int = 128, 
                       bg_color: Tuple[int, int, int] = None) -> np.ndarray:
        """
        Crop image based on mask and center it.
        
        Args:
            image: Input image
            mask: Binary mask
            output_size: Output image size (width, height)
            gray_bg: Whether to use gray background
            gray_value: Gray background value
            bg_color: Custom background color as RGB tuple (overrides gray_bg and gray_value)
            
        Returns:
            Cropped and centered image
        """
        # Find bounding box
        try:
            coords = np.where(mask > 0)
            if len(coords[0]) == 0:
                return None
                
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
        except Exception as e:
            logger.error(f"Error finding bounding box: {e}")
            return None
        
        # Crop image and mask
        cropped_image = image[y_min:y_max+1, x_min:x_max+1]
        cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]
        
        # Create output image with appropriate background
        if bg_color is not None:
            # Use custom background color
            output_image = np.full((output_size[1], output_size[0], 3), bg_color, dtype=np.uint8)
        else:
            # Use gray or black background
            output_image = np.full((output_size[1], output_size[0], 3), gray_value, dtype=np.uint8)
        
        # Calculate scaling to fit in output size
        h, w = cropped_image.shape[:2]
        scale = min(output_size[0] / w, output_size[1] / h)
        
        if scale < 1:
            # Resize to fit
            new_w, new_h = int(w * scale), int(h * scale)
            cropped_image = cv2.resize(cropped_image, (new_w, new_h))
            cropped_mask = cv2.resize(cropped_mask, (new_w, new_h))
            h, w = new_h, new_w
        
        # Center the image
        y_offset = (output_size[1] - h) // 2
        x_offset = (output_size[0] - w) // 2
        
        # Apply mask and place in output
        mask_3d = np.stack([cropped_mask] * 3, axis=2)
        
        if bg_color is not None:
            # Use custom background color
            output_image[y_offset:y_offset+h, x_offset:x_offset+w] = np.where(
                mask_3d > 0, 
                cropped_image, 
                bg_color
            )
        elif gray_bg:
            # Use gray background
            output_image[y_offset:y_offset+h, x_offset:x_offset+w] = np.where(
                mask_3d > 0, 
                cropped_image, 
                gray_value
            )
        else:
            # Use transparent background (black)
            output_image[y_offset:y_offset+h, x_offset:x_offset+w] = np.where(
                mask_3d > 0, 
                cropped_image, 
                0
            )
        
        return output_image
    
    def process_image(self, image_path: str, output_dir: str, 
                     min_area: int, max_area: int, 
                     padding: int = 10, hole_size: int = 5,
                     output_size: Tuple[int, int] = (1024, 1024),
                     gray_bg: bool = True, gray_value: int = 128,
                     bg_color: Tuple[int, int, int] = None, save_debug: bool = False,
                     sort_by_y: str = "ascending") -> int:
        """
        Process a single image and save cropped segments.
        
        Args:
            image_path: Path to input image
            output_dir: Output directory
            min_area: Minimum area threshold
            max_area: Maximum area threshold
            padding: Padding size
            hole_size: Hole removal size
            output_size: Output image size
            gray_bg: Whether to use gray background
            
        Returns:
            Number of segments saved
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return 0
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ensure image is in the correct format
        if len(image.shape) != 3 or image.shape[2] != 3:
            logger.error(f"Image must be RGB with 3 channels, got shape: {image.shape}")
            return 0
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        # Resize large images to prevent memory issues
        max_size = 1024
        h, w = image.shape[:2]
        if h > max_size or w > max_size:
            scale = min(max_size / h, max_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
            logger.info(f"Resized image from ({h}, {w}) to ({new_h}, {new_w})")
            
            # Save resized image for debugging if requested
            if save_debug:
                debug_dir = os.path.join(output_dir, "debug")
                os.makedirs(debug_dir, exist_ok=True)
                image_name = Path(image_path).stem
                debug_path = os.path.join(debug_dir, f"{image_name}_resized.png")
                debug_image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(debug_path, debug_image_bgr)
                logger.info(f"Saved resized debug image: {debug_path}")
        else:
            # Save original image for debugging if requested (when no resizing needed)
            if save_debug:
                debug_dir = os.path.join(output_dir, "debug")
                os.makedirs(debug_dir, exist_ok=True)
                image_name = Path(image_path).stem
                debug_path = os.path.join(debug_dir, f"{image_name}_original.png")
                debug_image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(debug_path, debug_image_bgr)
                logger.info(f"Saved original debug image: {debug_path}")
        
        # Generate masks with comprehensive error handling
        logger.info(f"Generating masks for: {image_path}")
        try:
            # Clear memory before generation
            if self.device == "cuda":
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            masks = self.mask_generator.generate(image)
            logger.info(f"Generated {len(masks)} masks")
            
        except (RuntimeError, IndexError) as e:
            error_msg = str(e)
            if "CUDA" in error_msg or "memory" in error_msg.lower():
                logger.error(f"CUDA/memory error during mask generation: {e}")
                logger.info("Trying with CPU fallback...")
                
                # Rebuild model on CPU
                from sam2.build_sam import build_sam2
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
                
                self.device = "cpu"
                self.model = build_sam2(
                    config_file=self.config_file, 
                    ckpt_path=self.ckpt_path, 
                    device="cpu", 
                    apply_postprocessing=False
                )
                self.mask_generator = SAM2AutomaticMaskGenerator(
                    model=self.model,
                    points_per_side=64,
                    points_per_batch=128,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=2,
                    crop_overlap_ratio=0.5,
                    min_mask_region_area=100,
                ) 
                
                masks = self.mask_generator.generate(image)
                logger.info(f"Generated {len(masks)} masks on CPU")
            elif "too many indices for tensor" in error_msg:
                logger.error(f"Tensor dimension error: {e}")
                logger.info("Trying with different mask generator settings...")
                
                # Try with different settings that might avoid the tensor issue
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
                self.mask_generator = SAM2AutomaticMaskGenerator(
                    model=self.model,
                    points_per_side=64,
                    points_per_batch=128,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=2,
                    crop_overlap_ratio=0.5,
                    min_mask_region_area=100,
                )
                
                masks = self.mask_generator.generate(image)
                logger.info(f"Generated {len(masks)} masks with alternative settings")
            else:
                logger.error(f"Error generating masks: {e}")
                logger.error(f"Image shape: {image.shape}, dtype: {image.dtype}")
                raise
        except Exception as e:
            logger.error(f"Error generating masks: {e}")
            logger.error(f"Image shape: {image.shape}, dtype: {image.dtype}")
            
            # Final fallback: try with minimal settings
            logger.info("Trying final fallback with minimal settings...")
            try:
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
                self.mask_generator = SAM2AutomaticMaskGenerator(
                    model=self.model,
                    points_per_side=64,
                    points_per_batch=128,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=2,
                    crop_overlap_ratio=0.5,
                    min_mask_region_area=100,
                )
                
                masks = self.mask_generator.generate(image)
                logger.info(f"Generated {len(masks)} masks with minimal settings")
            except Exception as fallback_error:
                logger.error(f"All mask generation attempts failed: {fallback_error}")
                raise
        
        # Filter by area
        filtered_masks = self.filter_masks_by_area(masks, min_area, max_area)
        
        # Remove background
        # filtered_masks = self.remove_background_mask(filtered_masks)
        
        if not filtered_masks:
            logger.warning(f"No valid segments found for: {image_path}")
            return 0
        
        # Sort masks by Y coordinate
        if sort_by_y in ["ascending", "descending"]:
            def get_mask_center_y(mask):
                """Get the Y coordinate of the mask center."""
                if isinstance(mask['segmentation'], dict):
                    try:
                        from pycocotools import mask as mask_utils
                        binary_mask = mask_utils.decode(mask['segmentation'])
                    except ImportError:
                        logger.error("pycocotools is not installed. Please install it with: pip install pycocotools")
                        raise
                else:
                    binary_mask = mask['segmentation'].astype(np.uint8)
                
                # Find the center Y coordinate
                coords = np.where(binary_mask > 0)
                if len(coords[0]) == 0:
                    return 0
                return np.mean(coords[0])
            
            reverse = (sort_by_y == "descending")
            filtered_masks = sorted(filtered_masks, key=get_mask_center_y, reverse=reverse)
            logger.info(f"Sorted {len(filtered_masks)} masks by Y coordinate ({sort_by_y} order)")
        
        # Create output directory
        image_name = Path(image_path).stem
        image_output_dir = os.path.join(output_dir, image_name)
        os.makedirs(image_output_dir, exist_ok=True)
        
        # Process each mask
        saved_count = 0
        for i, mask in enumerate(filtered_masks):
            try:
                # Process mask
                processed_mask = self.process_mask(mask, padding, hole_size)
                
                # Crop and center
                cropped_image = self.crop_and_center(
                    image, processed_mask, output_size, gray_bg, gray_value, bg_color
                )
                
                if cropped_image is not None:
                    # Save cropped image
                    output_path = os.path.join(image_output_dir, f"{image_name}_segment_{i:03d}.png")
                    cropped_image_bgr = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(output_path, cropped_image_bgr)
                    saved_count += 1
                    logger.info(f"Saved: {output_path}")
                else:
                    logger.warning(f"Could not crop mask {i}")
            except Exception as e:
                logger.error(f"Error processing mask {i}: {e}")
                continue
        
        # Clean up memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        logger.info(f"Saved {saved_count} segments for: {image_path}")
        return saved_count
    
    def process_directory(self, input_dir: str, output_dir: str, 
                         min_area: int, max_area: int, save_debug: bool = False, sort_by_y: str = "ascending", **kwargs) -> int:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Input directory containing images
            output_dir: Output directory
            min_area: Minimum area threshold
            max_area: Maximum area threshold
            **kwargs: Additional arguments for process_image
            
        Returns:
            Total number of segments saved
        """
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(f"*{ext}"))
            image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
        
        if not image_files:
            logger.error(f"No image files found in: {input_dir}")
            return 0
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each image
        total_segments = 0
        for image_file in image_files:
            try:
                segments = self.process_image(
                    str(image_file), output_dir, min_area, max_area, save_debug=save_debug, sort_by_y=sort_by_y, **kwargs
                )
                total_segments += segments
            except Exception as e:
                logger.error(f"Error processing {image_file}: {e}")
        
        logger.info(f"Total segments saved: {total_segments}")
        return total_segments

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="SAM2 Image Cropping Tool - Automatically crop image segments using SAM2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single image with default settings
  python sam2_crop_cli.py input.jpg output_dir/
  
  # Process a directory with custom area thresholds
  python sam2_crop_cli.py input_dir/ output_dir/ --min-area 1000 --max-area 50000
  
  # Use different output size
  python sam2_crop_cli.py input_dir/ output_dir/ --output-size 512 512
  
  # Process with custom padding and hole removal
  python sam2_crop_cli.py input_dir/ output_dir/ --padding 20 --hole-size 10
  
  # Use transparent background (black)
  python sam2_crop_cli.py input_dir/ output_dir/ --no-gray-bg
  
  # Use custom background color (red)
  python sam2_crop_cli.py input_dir/ output_dir/ --bg-color 255 0 0
  
  # Use custom gray value
  python sam2_crop_cli.py input_dir/ output_dir/ --gray-value 200
  
  # Process with debug images saved
  python sam2_crop_cli.py input_dir/ output_dir/ --save-debug
  
  # Sort segments by Y coordinate in descending order (top to bottom)
  python sam2_crop_cli.py input_dir/ output_dir/ --sort-by-y descending
  
  # Sort segments by Y coordinate in ascending order (bottom to top, default)
  python sam2_crop_cli.py input_dir/ output_dir/ --sort-by-y ascending
        """
    )
    
    # Input/Output arguments
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("output", help="Output directory")
    
    # Area threshold arguments
    parser.add_argument("--min-area", type=int, default=100, 
                       help="Minimum area threshold in pixels (default: 100)")
    parser.add_argument("--max-area", type=int, default=1000000, 
                       help="Maximum area threshold in pixels (default: 1000000)")
    
    # Model arguments
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu",
                       help="Device to run inference on (default: cpu)")
    parser.add_argument("--config-file", type=str, default="configs/sam2.1/sam2.1_hiera_b+.yaml",
                       help="Path to model config file (default: configs/sam2.1/sam2.1_hiera_b+.yaml)")
    parser.add_argument("--ckpt-path", type=str, default="checkpoints/sam2.1_hiera_base_plus.pt",
                       help="Path to model checkpoint file (default: checkpoints/sam2.1_hiera_base_plus.pt)")
    
    # Processing arguments
    parser.add_argument("--padding", type=int, default=10,
                       help="Padding size in pixels (default: 10)")
    parser.add_argument("--hole-size", type=int, default=5,
                       help="Size of holes to remove (default: 5)")
    parser.add_argument("--output-size", type=int, nargs=2, default=[1024, 1024],
                       metavar=("WIDTH", "HEIGHT"),
                       help="Output image size (default: 1024 1024)")
    parser.add_argument("--gray-bg", action="store_true", default=False,
                       help="Use gray background (default: True)")
    parser.add_argument("--no-gray-bg", dest="gray_bg", action="store_false",
                       help="Use transparent (black) background")
    parser.add_argument("--bg-color", type=int, nargs=3, metavar=("R", "G", "B"),
                       help="Custom background color as RGB values (0-255 each)")
    parser.add_argument("--gray-value", type=int, default=128,
                       help="Gray background value (0-255, default: 128)")
    
    # Sorting arguments
    parser.add_argument("--sort-by-y", choices=["ascending", "descending"], default="ascending",
                       help="Sort segments by Y coordinate (default: ascending)")
    
    # Other arguments
    parser.add_argument("--save-debug", action="store_true",
                       help="Save resized/original images for debugging")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Handle default gray background behavior
    # If neither --gray-bg nor --no-gray-bg is specified, default to gray background
    if not hasattr(args, 'gray_bg') or args.gray_bg is None:
        args.gray_bg = True
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not os.path.exists(args.input):
        logger.error(f"Input path does not exist: {args.input}")
        sys.exit(1)
    
    if args.min_area > args.max_area:
        logger.error("min-area cannot be greater than max-area")
        sys.exit(1)
    
    if args.gray_value < 0 or args.gray_value > 255:
        logger.error("gray-value must be between 0 and 255")
        sys.exit(1)
    
    # Validate bg-color if provided
    if args.bg_color is not None:
        for i, val in enumerate(args.bg_color):
            if val < 0 or val > 255:
                logger.error(f"bg-color value {i+1} ({val}) must be between 0 and 255")
                sys.exit(1)
    
    try:
        # Initialize SAM2 cropper
        cropper = SAM2Cropper(
            device=args.device,
            config_file=args.config_file,
            ckpt_path=args.ckpt_path
        )
        
        # Process input
        if os.path.isfile(args.input):
            # Single image
            segments = cropper.process_image(
                args.input, args.output, args.min_area, args.max_area,
                padding=args.padding, hole_size=args.hole_size,
                output_size=tuple(args.output_size), gray_bg=args.gray_bg,
                gray_value=args.gray_value, bg_color=args.bg_color,
                save_debug=args.save_debug, sort_by_y=args.sort_by_y
            )
            logger.info(f"Processing complete. Saved {segments} segments.")
        else:
            # Directory
            segments = cropper.process_directory(
                args.input, args.output, args.min_area, args.max_area,
                padding=args.padding, hole_size=args.hole_size,
                output_size=tuple(args.output_size), gray_bg=args.gray_bg,
                gray_value=args.gray_value, bg_color=args.bg_color,
                save_debug=args.save_debug, sort_by_y=args.sort_by_y
            )
            logger.info(f"Processing complete. Saved {segments} total segments.")
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 