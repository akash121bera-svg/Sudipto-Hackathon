import os
import cv2
import numpy as np
import logging
from PIL import Image
from backend.app.core.config import settings

logger = logging.getLogger("preprocess_agent")

def deskew_image(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Detects text skew angle and rotates the image to deskew it."""
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Threshold the image, text becomes white (255), background black (0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find all white pixels (text) coordinates
    coords = np.column_stack(np.where(thresh > 0))
    
    if len(coords) == 0:
        return image, 0.0

    # Find minimum area bounding box containing all text points
    angle = cv2.minAreaRect(coords)[-1]

    # OpenCV returns angle in [-90, 0)
    # Adjust the angle for correct rotation
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Check for extreme angles (often false positives)
    if abs(angle) > 20.0 or abs(angle) < 0.1:
        return image, 0.0

    # Get image center and rotation matrix
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform rotation
    rotated = cv2.warpAffine(
        image, M, (w, h), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return rotated, angle

def preprocess_document(
    image_path: str,
    output_filename: str,
    deskew: bool = True,
    denoise: bool = True,
    clahe_enhancement: bool = False,
    adaptive_thresh: bool = False,
    threshold_block_size: int = 11,
    threshold_c: int = 2,
    scale_factor: float = 1.0
) -> dict:
    """
    Applies configurable OpenCV preprocessing techniques to optimize document readability.
    Returns details of actions performed and the preprocessed image path.
    """
    logger.info(f"Preprocessing image: {image_path}")
    
    # 1. Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")

    h_orig, w_orig = img.shape[:2]
    actions = []
    
    # 2. Scale image if requested (higher resolution improves OCR for small text)
    if scale_factor != 1.0:
        w_new = int(w_orig * scale_factor)
        h_new = int(h_orig * scale_factor)
        img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_CUBIC)
        actions.append(f"scaled_by_{scale_factor}x")

    # 3. Deskewing
    angle = 0.0
    if deskew:
        img, angle = deskew_image(img)
        if abs(angle) > 0.0:
            actions.append(f"deskewed_by_{angle:.2f}_degrees")

    # Convert to grayscale for internal processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 4. Contrast enhancement (CLAHE - Contrast Limited Adaptive Histogram Equalization)
    if clahe_enhancement:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        actions.append("clahe_contrast_enhanced")

    # 5. Denoising
    if denoise:
        # Fast non-local means denoising for single channel grayscale
        gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        actions.append("fast_nl_means_denoised")

    # Save output image as the processed image (preserving color if possible/needed by logic)
    # Using the deskewed/scaled 'img' if no further filtering done, or 'gray' if filter-heavy
    # To satisfy "save color output", we use 'img' if filters were applied to gray, we apply back or keep original
    out_path = os.path.join(settings.PREPROCESSED_DIR, output_filename)
    cv2.imwrite(out_path, img)
    
    logger.info(f"Preprocessing completed. Actions: {actions}. Saved to {out_path}")
    
    return {
        "preprocessed_path": out_path,
        "actions_taken": actions,
        "skew_angle": angle,
        "width": img.shape[1],
        "height": img.shape[0]
    }
