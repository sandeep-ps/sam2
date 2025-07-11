#!/usr/bin/env python3
"""
Minimal SAM2 test script to isolate issues with the model.
"""

import sys
import logging
import cv2
import numpy as np
import torch
import os
import gc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_cuda_environment():
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

def safe_device_selection():
    """Safely select device with fallback to CPU."""
    try:
        if torch.cuda.is_available():
            # Check if CUDA memory is sufficient
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory < 4.0:  # Less than 4GB
                logger.warning(f"GPU memory ({gpu_memory:.1f}GB) may be insufficient, using CPU")
                return "cpu"
            return "cuda"
        else:
            return "cpu"
    except Exception as e:
        logger.warning(f"Device selection failed: {e}, using CPU")
        return "cpu"

def test_sam2_minimal():
    """Test SAM2 with minimal setup and CUDA error handling."""
    
    try:
        # Setup environment
        setup_cuda_environment()
        device = safe_device_selection()
        logger.info(f"Using device: {device}")
        
        # Import SAM2 modules
        logger.info("Importing SAM2 modules...")
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        
        # Test 1: Build model with error handling
        logger.info("Testing model building...")
        try:
            model = build_sam2("sam2_hiera_b+", device=device)
            logger.info("✓ Model built successfully")
        except Exception as e:
            logger.error(f"Model building failed on {device}: {e}")
            if device == "cuda":
                logger.info("Falling back to CPU...")
                device = "cpu"
                model = build_sam2("sam2_hiera_b+", device=device)
                logger.info("✓ Model built successfully on CPU")
            else:
                raise
        
        # Test 2: Create mask generator with conservative settings
        logger.info("Testing mask generator creation...")
        mask_generator = SAM2AutomaticMaskGenerator(
            model=model,
            points_per_side=16,  # Reduced from 32
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        logger.info("✓ Mask generator created successfully")
        
        # Test 3: Create a simple test image (smaller size)
        logger.info("Creating test image...")
        test_image = np.zeros((256, 256, 3), dtype=np.uint8)  # Reduced from 512x512
        # Add a simple rectangle
        test_image[64:192, 64:192] = [255, 255, 255]
        logger.info(f"✓ Test image created: shape={test_image.shape}, dtype={test_image.dtype}")
        
        # Test 4: Generate masks with error handling
        logger.info("Testing mask generation...")
        try:
            # Clear memory before generation
            if device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            masks = mask_generator.generate(test_image)
            logger.info(f"✓ Generated {len(masks)} masks")
            
        except RuntimeError as e:
            if "CUDA" in str(e) or "memory" in str(e).lower():
                logger.error(f"CUDA error during mask generation: {e}")
                logger.info("Trying with CPU fallback...")
                
                # Rebuild model on CPU
                model = build_sam2("sam2_hiera_b+", device="cpu")
                mask_generator = SAM2AutomaticMaskGenerator(
                    model=model,
                    points_per_side=16,
                    pred_iou_thresh=0.86,
                    stability_score_thresh=0.92,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=100,
                )
                
                masks = mask_generator.generate(test_image)
                logger.info(f"✓ Generated {len(masks)} masks on CPU")
            else:
                raise
        
        # Test 5: Process masks
        logger.info("Testing mask processing...")
        for i, mask in enumerate(masks[:3]):  # Test first 3 masks
            logger.info(f"  Mask {i}: area={mask['area']}, bbox={mask['bbox']}")
        
        # Clean up
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("✓ All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_image(image_path):
    """Test with a real image with enhanced error handling."""
    
    try:
        logger.info(f"Testing with real image: {image_path}")
        
        # Setup environment
        device = safe_device_selection()
        logger.info(f"Using device: {device}")
        
        # Import SAM2 modules
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return False
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logger.info(f"Loaded image: shape={image.shape}, dtype={image.dtype}")
        
        # Resize if too large (more conservative)
        max_size = 384  # Reduced from 512
        h, w = image.shape[:2]
        if h > max_size or w > max_size:
            scale = min(max_size / h, max_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
            logger.info(f"Resized to: ({new_h}, {new_w})")
        
        # Build model with fallback
        logger.info("Building model...")
        try:
            model = build_sam2("sam2_hiera_b+", device=device)
        except Exception as e:
            logger.error(f"Model building failed on {device}: {e}")
            device = "cpu"
            model = build_sam2("sam2_hiera_b+", device=device)
            logger.info("✓ Model built successfully on CPU")
        
        # Create mask generator with conservative settings
        logger.info("Creating mask generator...")
        mask_generator = SAM2AutomaticMaskGenerator(
            model=model,
            points_per_side=12,  # Further reduced
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        
        # Generate masks with error handling
        logger.info("Generating masks...")
        try:
            if device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            masks = mask_generator.generate(image)
            logger.info(f"✓ Generated {len(masks)} masks")
            
        except RuntimeError as e:
            if "CUDA" in str(e) or "memory" in str(e).lower():
                logger.error(f"CUDA error: {e}")
                logger.info("Retrying with CPU...")
                
                model = build_sam2("sam2_hiera_b+", device="cpu")
                mask_generator = SAM2AutomaticMaskGenerator(
                    model=model,
                    points_per_side=12,
                    pred_iou_thresh=0.86,
                    stability_score_thresh=0.92,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=100,
                )
                
                masks = mask_generator.generate(image)
                logger.info(f"✓ Generated {len(masks)} masks on CPU")
            else:
                raise
        
        # Show mask info
        for i, mask in enumerate(masks[:5]):  # Show first 5 masks
            logger.info(f"  Mask {i}: area={mask['area']}, bbox={mask['bbox']}")
        
        # Clean up
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Real image test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_models():
    """Test different SAM2 models with error handling."""
    
    # Import at function level
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    
    models = ["sam2_hiera_t", "sam2_hiera_s", "sam2_hiera_l", "sam2_hiera_b+"]  # Start with smallest
    
    for model_name in models:
        try:
            logger.info(f"Testing model: {model_name}")
            
            # Use CPU for testing to avoid CUDA issues
            device = "cpu"
            
            # Build model
            model = build_sam2(model_name, device=device)
            logger.info(f"✓ {model_name} built successfully")
            
            # Create simple test image (smaller)
            test_image = np.zeros((128, 128, 3), dtype=np.uint8)  # Reduced size
            test_image[32:96, 32:96] = [255, 255, 255]
            
            # Create mask generator with very conservative settings
            mask_generator = SAM2AutomaticMaskGenerator(
                model=model,
                points_per_side=8,  # Very conservative
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=50,  # Reduced
            )
            
            # Generate masks
            masks = mask_generator.generate(test_image)
            logger.info(f"✓ {model_name} generated {len(masks)} masks")
            
            # Clean up
            del model, mask_generator
            gc.collect()
            
        except Exception as e:
            logger.error(f"✗ {model_name} failed: {e}")

if __name__ == "__main__":
    logger.info("Starting SAM2 minimal tests with CUDA error handling...")
    
    # Test 1: Minimal functionality
    logger.info("\n=== Test 1: Minimal Functionality ===")
    success1 = test_sam2_minimal()
    
    # Test 2: Different models
    logger.info("\n=== Test 2: Different Models ===")
    test_different_models()
    
    # Test 3: Real image (if available)
    logger.info("\n=== Test 3: Real Image ===")
    test_image_path = "notebooks/images/cars.jpg"
    if success1:
        test_with_real_image(test_image_path)
    
    logger.info("\n=== Test Summary ===")
    if success1:
        logger.info("✓ Basic SAM2 functionality works")
        logger.info("If the CLI still fails, the issue is in the CLI code, not the model.")
    else:
        logger.info("✗ SAM2 model has issues")
        logger.info("The problem is with the SAM2 installation or model weights.") 