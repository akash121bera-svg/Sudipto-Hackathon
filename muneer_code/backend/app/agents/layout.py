import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from backend.app.core.vector_store import search_embeddings

logger = logging.getLogger("layout_agent")

# ─── Predefined templates ────────────────────────────────────────────────────
# Coordinates are normalized (0 to 1000).  These are the ORIGINAL coarse boxes,
# kept only as a last-resort fallback if line detection fails.
STANDARD_TEMPLATES = {
    "membership_application": {
        "name": "Standard Membership Application",
        "fields": [
            {"key": "full_name", "type": "printed_handwritten", "bbox": [150, 100, 250, 900], "label": "Full Name"},
            {"key": "date_of_birth", "type": "printed_handwritten", "bbox": [280, 100, 350, 480], "label": "Date of Birth"},
            {"key": "phone_number", "type": "printed_handwritten", "bbox": [280, 520, 350, 900], "label": "Phone Number"},
            {"key": "email_address", "type": "printed_handwritten", "bbox": [380, 100, 450, 900], "label": "Email"},
            {"key": "membership_type", "type": "checkbox", "bbox": [480, 100, 540, 900], "options": [
                {"label": "Standard", "bbox": [480, 200, 520, 350]},
                {"label": "Premium", "bbox": [480, 400, 520, 550]},
                {"label": "Student", "bbox": [480, 600, 520, 750]}
            ]},
            {"key": "signature", "type": "handwriting", "bbox": [700, 500, 800, 900], "label": "Signature"}
        ]
    },
    # The Namibian template now uses refined, label-excluding coordinates.
    # bbox = [ymin, xmin, ymax, xmax] in normalized 0-1000 space.
    # These are VALUE-ONLY regions (label text has been cropped out).
    "namibian_employment_application": {
        "name": "Namibian Public Service Employment Application",
        "fields": [
            # Bboxes precisely calibrated from pixel row survey on 768x889 reference scan.
            # Format: [ymin, xmin, ymax, xmax] normalized to 0-1000 scale.
            # ymin is pushed DOWN past the printed label line to capture only the handwritten value.
            # xmin is pushed RIGHT past inline labels (Citizenship:, Home No.:, Email:).
            {"key": "position_applied",     "type": "printed_handwritten", "bbox": [240,  13, 270, 490], "label": "Position Applied For"},
            {"key": "office_ministry",      "type": "printed_handwritten", "bbox": [240, 495, 295, 990], "label": "Office/Ministry/Agency"},
            {"key": "duty_station",         "type": "printed_handwritten", "bbox": [303,  13, 338, 490], "label": "Duty Station"},
            {"key": "assume_duty",          "type": "printed_handwritten", "bbox": [303, 495, 338, 990], "label": "When can you assume duty?"},
            {"key": "full_name",            "type": "printed_handwritten", "bbox": [429, 234, 467, 723], "label": "Full Name (First Names)"},
            {"key": "namibian_id",          "type": "printed_handwritten", "bbox": [467,  13, 503, 352], "label": "Namibian ID Number"},
            {"key": "date_of_birth",        "type": "printed_handwritten", "bbox": [467, 430, 503, 723], "label": "Date of Birth"},
            {"key": "citizenship",          "type": "printed_handwritten", "bbox": [515, 185, 538, 352], "label": "Citizenship"},
            {"key": "postal_address",       "type": "printed_handwritten", "bbox": [540,  13, 590, 352], "label": "Postal Address"},
            {"key": "residential_address",  "type": "printed_handwritten", "bbox": [540, 430, 590, 723], "label": "Residential Address"},
            {"key": "phone_number",         "type": "printed_handwritten", "bbox": [618, 145, 640, 330], "label": "Phone Number"},
            {"key": "email_address",        "type": "printed_handwritten", "bbox": [638,  90, 660, 660], "label": "Email Address"},
        ]
    }
}

# ─── Label keyword map for dynamic extraction ────────────────────────────────
# Each key maps to label keywords that EasyOCR may detect (including common
# misreadings from low-quality scans).
LABEL_KEYWORDS = {
    "position_applied":    ["position", "applied", "toltion"],
    "office_ministry":     ["office", "ministry", "agency", "dept", "kencv"],
    "duty_station":        ["duty", "station", "dunyaullon"],
    "assume_duty":         ["assume", "youjrume"],
    "full_name":           ["names", "block", "nrnes", "suramc", "fwoinard"],
    "namibian_id":         ["identity", "nuni", "nid"],
    "date_of_birth":       ["birth", "oato"],
    "citizenship":         ["citizenship", "citllenthle"],
    "postal_address":      ["postal", "foxtul"],
    "residential_address": ["residential", "jdres"],
    "phone_number":        ["phone", "mobile", "moulle", "contact"],
    "email_address":       ["email", "e-mail", "enim"],
}


# ─── Visual classification ───────────────────────────────────────────────────

def is_namibian_form_visually(img: np.ndarray) -> bool:
    """
    Checks if the image has a dark emblem in the top-middle region (Namibian Coat of Arms).
    """
    try:
        h, w = img.shape[:2]
        # Crop top-middle area: y from 2% to 18%, x from 38% to 62%
        ymin, ymax = int(h * 0.02), int(h * 0.18)
        xmin, xmax = int(w * 0.38), int(w * 0.62)
        
        crop = img[ymin:ymax, xmin:xmax]
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
            
        # Threshold to binary
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Count non-zero (black) pixels density in binarized patch
        density = np.sum(thresh == 255) / thresh.size
        
        # Count contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        logger.info(f"Visual check for Namibian emblem: density={density:.3f}, contours={len(contours)}")
        return density > 0.07 and len(contours) > 12
    except Exception as e:
        logger.warning(f"Visual emblem detection failed: {e}")
        return False


# ─── Horizontal line detection ───────────────────────────────────────────────

def detect_horizontal_lines(img: np.ndarray, min_density: float = 0.2) -> List[int]:
    """
    Detects horizontal ruled lines in a form using morphological operations.
    Returns a list of y-coordinates (pixel rows) where horizontal lines exist.
    
    These lines are used to snap bounding box top/bottom edges to actual
    form row boundaries, compensating for scan rotation/shift.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cols = thresh.shape[1]

    # Long horizontal structuring element — captures lines spanning ≥10% of page width
    horizontal_size = cols // 10
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)

    # Find rows with significant horizontal line coverage
    row_sums = np.sum(horizontal_lines > 0, axis=1)
    line_rows = np.where(row_sums > (cols * min_density))[0]

    if len(line_rows) == 0:
        return []

    # Group adjacent rows into single line positions (take mean of each group)
    groups = np.split(line_rows, np.where(np.diff(line_rows) > 5)[0] + 1)
    detected = [int(np.mean(g)) for g in groups if len(g) > 0]

    logger.info(f"Detected {len(detected)} horizontal lines at y={detected}")
    return detected


def snap_y_to_lines(y: int, lines: List[int], tolerance: int = 25, 
                    snap_mode: str = "after") -> int:
    """
    Snaps a y-coordinate to the nearest horizontal line within tolerance.
    
    snap_mode:
      "after"  — places y just below the line (for top edges: y = line + 3)
      "before" — places y just above the line (for bottom edges: y = line - 3)
    """
    nearest = [ly for ly in lines if abs(ly - y) <= tolerance]
    if not nearest:
        return y
    
    best = min(nearest, key=lambda ly: abs(ly - y))
    if snap_mode == "after":
        return best + 3
    else:
        return best - 3


# ─── Checkbox detection ──────────────────────────────────────────────────────

def detect_checkboxes_opencv(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Uses OpenCV contour analysis to find square boxes that look like form checkboxes.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Detect horizontal and vertical lines to find square boxes
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    checkboxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            # Checkboxes are small, square-like elements
            if 0.8 <= aspect_ratio <= 1.2 and 15 <= w <= 50 and 15 <= h <= 50:
                checkboxes.append((x, y, w, h))
                
    return checkboxes


# ─── Dynamic label-anchored extraction ───────────────────────────────────────

def find_label_anchor(full_ocr_results: list, keywords: List[str]) -> Optional[Tuple[int, int, int, int]]:
    """
    Searches full-image OCR results for a label matching any of the given keywords.
    Returns the label's bounding box as (y1, x1, y2, x2) or None.
    """
    for bbox_pts, text, conf in full_ocr_results:
        text_lower = text.lower()
        for kw in keywords:
            if kw in text_lower:
                pts = np.array(bbox_pts)
                y1 = int(pts[:, 1].min())
                y2 = int(pts[:, 1].max())
                x1 = int(pts[:, 0].min())
                x2 = int(pts[:, 0].max())
                return (y1, x1, y2, x2)
    return None


def build_dynamic_fields(img: np.ndarray, full_ocr_results: list) -> List[Dict[str, Any]]:
    """
    Builds field definitions by finding label anchors in OCR results and
    creating value crop boxes to the right of each label.
    
    This is the most robust strategy for handling scanned forms with
    rotation/shift, since it doesn't rely on absolute pixel coordinates.
    """
    h_img, w_img = img.shape[:2]
    fields = []

    for field_key, keywords in LABEL_KEYWORDS.items():
        anchor = find_label_anchor(full_ocr_results, keywords)
        if anchor is None:
            logger.debug(f"Dynamic extraction: label for '{field_key}' not found")
            continue

        ly1, lx1, ly2, lx2 = anchor

        # Value crop region: same vertical band as label, extending to the right
        vy1 = max(0, ly1 - 10)
        vy2 = min(h_img, ly2 + 10)
        vx1 = lx2 + 5  # Start just after the label ends

        # Determine crop width based on expected column layout
        # Left-column fields don't extend past the page center
        is_left_col = field_key in [
            "position_applied", "duty_station", "namibian_id",
            "citizenship", "postal_address", "phone_number"
        ]
        if is_left_col:
            vx2 = min(w_img, lx2 + 250)
        else:
            vx2 = min(w_img, lx2 + 350)

        # Ensure minimum width
        if vx2 < vx1 + 50:
            vx2 = min(w_img, vx1 + 200)

        fields.append({
            "key": field_key,
            "type": "printed_handwritten",
            "bbox": [vy1, vx1, vy2, vx2],
            "label": LABEL_KEYWORDS.get(field_key, [field_key])[0].title(),
            "extraction_method": "label_anchored"
        })

    logger.info(f"Dynamic label extraction found {len(fields)} fields")
    return fields


# ─── Main layout analysis ────────────────────────────────────────────────────

def analyze_layout(image_path: str, document_text_sample: str = "") -> Dict[str, Any]:
    """
    Analyzes document layout using a 3-tier strategy:
    
    1. Template matching (visual + text checks) to identify the form type
    2. Horizontal line detection to snap bounding boxes to actual form rows
    3. Refined calibrated bounding boxes that exclude label text
    
    The result includes field definitions with absolute pixel bounding boxes
    that the OCR agent can use to crop and extract values.
    """
    logger.info(f"Analyzing layout for {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    h_img, w_img = img.shape[:2]
    
    # ── Step 1: Template classification ───────────────────────────────────
    matched_template_key = None
    template_confidence = 0.0
    
    # 1a. Visual check for Namibian Coat of Arms (highest priority)
    if is_namibian_form_visually(img):
        matched_template_key = "namibian_employment_application"
        template_confidence = 0.95
        logger.info("Matched template visually: namibian_employment_application")
        
    # 1b. Filename/text checks if visual check was negative/indeterminate
    if not matched_template_key and document_text_sample:
        text_lower = document_text_sample.lower()
        if "namibia" in text_lower or "ferdinard" in text_lower or "employment" in text_lower or "public" in text_lower:
            matched_template_key = "namibian_employment_application"
            template_confidence = 0.90
        elif "membership" in text_lower:
            matched_template_key = "membership_application"
            template_confidence = 0.90
        elif "sample" in text_lower:
            # Default to membership application for generic sample naming
            matched_template_key = "membership_application"
            template_confidence = 0.80
            
        if not matched_template_key:
            # Search Qdrant for similar template descriptions or keywords
            try:
                results = search_embeddings("form_templates", document_text_sample, limit=1)
                if results and results[0]["score"] > 0.6:
                    matched_template_key = results[0]["payload"]["template_key"]
                    template_confidence = results[0]["score"]
                    logger.info(f"Matched Qdrant template: {matched_template_key} (score: {template_confidence})")
            except Exception as e:
                logger.warning(f"Qdrant template search failed: {e}. Falling back to default match.")
            
    # Default fallback if still unmatched
    if not matched_template_key and document_text_sample:
        if "membership" in document_text_sample.lower() or "application" in document_text_sample.lower():
            matched_template_key = "membership_application"
            template_confidence = 0.75
        
    # ── Step 2: Detect horizontal lines for bbox snapping ─────────────────
    h_lines = detect_horizontal_lines(img)
    
    # ── Step 3: Build field definitions ───────────────────────────────────
    fields = []
    
    if matched_template_key and matched_template_key in STANDARD_TEMPLATES:
        template = STANDARD_TEMPLATES[matched_template_key]
        logger.info(f"Applying template fields for: {template['name']}")
        
        # Convert normalized template coordinates to absolute pixel dimensions
        # then snap to detected horizontal lines for better accuracy
        for f in template["fields"]:
            # bbox is in [y_min, x_min, y_max, x_max] normalized from 0 to 1000
            ymin, xmin, ymax, xmax = f["bbox"]
            abs_ymin = int(ymin * h_img / 1000)
            abs_xmin = int(xmin * w_img / 1000)
            abs_ymax = int(ymax * h_img / 1000)
            abs_xmax = int(xmax * w_img / 1000)
            
            # Snap y-coordinates to detected horizontal lines
            if h_lines:
                abs_ymin = snap_y_to_lines(abs_ymin, h_lines, tolerance=25, snap_mode="after")
                abs_ymax = snap_y_to_lines(abs_ymax, h_lines, tolerance=25, snap_mode="before")
                
                # Safety: ensure ymax > ymin after snapping (min 25px height for OCR)
                if abs_ymax <= abs_ymin + 25:
                    # Revert to unsnapped values — the snapping squished this row
                    abs_ymin = int(ymin * h_img / 1000)
                    abs_ymax = int(ymax * h_img / 1000)
            
            abs_bbox = [abs_ymin, abs_xmin, abs_ymax, abs_xmax]
            
            field_data = {
                "key": f["key"],
                "type": f["type"],
                "bbox": abs_bbox,
                "label": f.get("label", f["key"]),
                "extraction_method": "template_line_snapped" if h_lines else "template_normalized"
            }
            
            # Handle checkboxes options coordinates mapping
            if f["type"] == "checkbox" and "options" in f:
                opt_list = []
                for opt in f["options"]:
                    oymin, oxmin, oymax, oxmax = opt["bbox"]
                    o_bbox = [
                        int(oymin * h_img / 1000),
                        int(oxmin * w_img / 1000),
                        int(oymax * h_img / 1000),
                        int(oxmax * w_img / 1000)
                    ]
                    opt_list.append({"label": opt["label"], "bbox": o_bbox})
                field_data["options"] = opt_list
                
            fields.append(field_data)
    else:
        # Dynamic contour-based layout segmentation (No template match)
        logger.info("No template match. Auto-generating layout regions using OpenCV.")
        detected_checkboxes = detect_checkboxes_opencv(img)
        
        # Add auto-detected boxes
        for i, box in enumerate(detected_checkboxes):
            x, y, w, h = box
            fields.append({
                "key": f"checkbox_field_{i}",
                "type": "checkbox",
                "bbox": [y, x, y + h, x + w],
                "label": f"Detected Box {i+1}"
            })
            
        # Add generic page zones for main text blocks
        # Simply split the page into three horizontal blocks: Top, Middle, Bottom
        fields.append({
            "key": "top_section",
            "type": "printed_handwritten",
            "bbox": [0, 0, int(h_img * 0.33), w_img],
            "label": "Top Section"
        })
        fields.append({
            "key": "middle_section",
            "type": "printed_handwritten",
            "bbox": [int(h_img * 0.33), 0, int(h_img * 0.66), w_img],
            "label": "Middle Section"
        })
        fields.append({
            "key": "bottom_section",
            "type": "printed_handwritten",
            "bbox": [int(h_img * 0.66), 0, h_img, w_img],
            "label": "Bottom Section"
        })

    return {
        "template_key": matched_template_key or "generic_layout",
        "template_confidence": template_confidence,
        "fields": fields,
        "horizontal_lines": h_lines,
        "image_dimensions": {"width": w_img, "height": h_img}
    }
