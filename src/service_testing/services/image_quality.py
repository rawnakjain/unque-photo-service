"""Image quality validation utilities for blur and corruption detection."""

import io
import logging
import time
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ImageQualityError(ValueError):
    """Raised when image fails quality checks."""
    pass


def validate_image_quality(
    image_bytes: bytes,
    blur_threshold: float = 100.0,
    min_dimension: int = 50,
    max_dimension: int = 10000,
) -> Tuple[bool, str]:
    """
    Validate image quality: check for corruption, blur, and reasonable dimensions.
    
    Args:
        image_bytes: Raw image bytes
        blur_threshold: Laplacian variance threshold for blur detection (lower = more blur)
        min_dimension: Minimum width/height in pixels
        max_dimension: Maximum width/height in pixels
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    _t_start = time.perf_counter()

    # Check 1: Basic corruption - can PIL open it?
    _t0 = time.perf_counter()
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image.verify()  # Verify integrity
    except Exception as e:
        logger.warning(f"Image corruption detected: {e}")
        return False, "Image is corrupted or invalid"
    _t_verify = (time.perf_counter() - _t0) * 1000

    # Re-open for further checks (verify() consumes the stream)
    _t0 = time.perf_counter()
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image.load()
        width, height = pil_image.size
    except Exception as e:
        logger.warning(f"Image dimension check failed: {e}")
        return False, "Image is corrupted or invalid"
    _t_reload = (time.perf_counter() - _t0) * 1000

    # Check 2: Dimension bounds
    if width < min_dimension or height < min_dimension:
        logger.warning(f"Image too small: {width}x{height}")
        return False, f"Image dimensions too small (minimum {min_dimension}x{min_dimension}px)"

    if width > max_dimension or height > max_dimension:
        logger.warning(f"Image too large: {width}x{height}")
        return False, f"Image dimensions too large (maximum {max_dimension}x{max_dimension}px)"

    # Check 3: Blur detection using OpenCV Laplacian variance
    _t0 = time.perf_counter()
    try:
        # Normalize grayscale, RGB, and alpha-channel images before OpenCV conversion.
        rgb_image = pil_image.convert("RGB")
        cv_image = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance (focus measure)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        logger.info(f"Image blur score (Laplacian variance): {laplacian_var:.2f}")
        
        if laplacian_var < blur_threshold:
            logger.warning(f"Image is blurred: {laplacian_var:.2f} < {blur_threshold}")
            return False, "Image is blurred"
            
    except Exception as e:
        logger.warning(f"Blur detection failed: {e}")
        # Don't fail on blur detection errors, just log and continue
        pass
    _t_blur = (time.perf_counter() - _t0) * 1000

    logger.info(
        "[stage-timing] image_validation: verify=%.1fms reload=%.1fms blur=%.1fms total=%.1fms",
        _t_verify,
        _t_reload,
        _t_blur,
        (time.perf_counter() - _t_start) * 1000,
    )

    return True, ""


def check_image_quality_or_raise(image_bytes: bytes, **kwargs) -> None:
    """
    Check image quality and raise ImageQualityError if invalid.
    
    Args:
        image_bytes: Raw image bytes
        **kwargs: Additional arguments passed to validate_image_quality
    
    Raises:
        ImageQualityError: If image fails quality checks
    """
    is_valid, error_message = validate_image_quality(image_bytes, **kwargs)
    if not is_valid:
        raise ImageQualityError(error_message)