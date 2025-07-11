#!/usr/bin/env python3
"""
Test script to verify the CLI fix for tensor dimension errors.
"""

import sys
import logging
import cv2
import numpy as np
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_cli_fix():
    """Test the CLI fix for tensor dimension errors."""
    
    try:
        # Import the CLI class
        from sam2_crop_cli import SAM2Cropper
        
        # Create a simple test image
        test_image = np.zeros((256, 256, 3), dtype=np.uint8)
        test_image[64:192, 64:192] = [255, 255, 255]
        
        # Save test image
        test_image_path = "test_image.jpg"
        cv2.imwrite(test_image_path, cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
        
        # Initialize cropper
        logger.info("Initializing SAM2Cropper...")
        cropper = SAM2Cropper(
            model_type="sam2_hiera_t", 
            device="cpu"
        )  # Use smallest model with default config/checkpoint
        
        # Test processing
        logger.info("Testing image processing...")
        output_dir = "test_output"
        segments = cropper.process_image(
            test_image_path, 
            output_dir, 
            min_area=100, 
            max_area=10000,
            padding=5,
            hole_size=3,
            output_size=(512, 512),
            gray_bg=True
        )
        
        logger.info(f"✓ Successfully processed image, saved {segments} segments")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Testing CLI fix for tensor dimension errors...")
    success = test_cli_fix()
    
    if success:
        logger.info("✓ CLI fix test passed!")
    else:
        logger.info("✗ CLI fix test failed!")
        sys.exit(1) 