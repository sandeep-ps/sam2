#!/usr/bin/env python3
"""
Test script to check if all dependencies are working properly.
"""

import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dependencies():
    """Test all required dependencies."""
    
    # Test numpy
    try:
        import numpy as np
        logger.info("✓ NumPy imported successfully")
        logger.info(f"  NumPy version: {np.__version__}")
        
        # Test basic numpy operations
        arr = np.array([1, 2, 3])
        logger.info("✓ NumPy basic operations work")
        
    except ImportError as e:
        logger.error(f"✗ NumPy import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ NumPy operations failed: {e}")
        return False
    
    # Test OpenCV
    try:
        import cv2
        logger.info("✓ OpenCV imported successfully")
        logger.info(f"  OpenCV version: {cv2.__version__}")
        
        # Test basic OpenCV operations
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        logger.info("✓ OpenCV basic operations work")
        
    except ImportError as e:
        logger.error(f"✗ OpenCV import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ OpenCV operations failed: {e}")
        return False
    
    # Test PyTorch
    try:
        import torch
        logger.info("✓ PyTorch imported successfully")
        logger.info(f"  PyTorch version: {torch.__version__}")
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
        
        # Test basic PyTorch operations
        tensor = torch.tensor([1, 2, 3])
        logger.info("✓ PyTorch basic operations work")
        
    except ImportError as e:
        logger.error(f"✗ PyTorch import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ PyTorch operations failed: {e}")
        return False
    
    # Test SAM2
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.utils.amg import remove_small_regions
        logger.info("✓ SAM2 modules imported successfully")
        
    except ImportError as e:
        logger.error(f"✗ SAM2 import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ SAM2 operations failed: {e}")
        return False
    
    # Test pycocotools (optional)
    try:
        import pycocotools
        logger.info("✓ pycocotools imported successfully")
    except ImportError:
        logger.warning("⚠ pycocotools not available (optional dependency)")
    
    logger.info("All dependency tests passed!")
    return True

if __name__ == "__main__":
    success = test_dependencies()
    if not success:
        sys.exit(1) 