"""
ocr_engine.py
─────────────
Real field-level OCR engine for Document Intelligence Agent.

Priority chain (per-field extraction):
  1. Gemini Vision  (best quality — handles scanned handwriting superbly)
  2. EasyOCR        (deep-learning, no binary required, good for printed text)
  3. pytesseract    (if Tesseract binary is available)
  4. Mock fallback  (demo-safe, never crashes)

The Gemini engine works on the FULL FORM IMAGE and extracts all field values
in one structured API call, which gives it full document context.
Set GEMINI_API_KEY in .env to enable it.

Field preprocessing is tuned per field type for the EasyOCR fallback:
  - printed_handwritten → CLAHE + gentle denoise + Otsu binarize
  - handwriting         → aggressive CLAHE + adaptive threshold + deskew
  - checkbox            → pixel-density check, no OCR
"""

import os
import io
import cv2
import json
import logging
import numpy as np
from typing import Tuple, Optional, Dict, Any
from PIL import Image

logger = logging.getLogger("ocr_engine")


# ─── Gemini Vision OCR ────────────────────────────────────────────────────────

_gemini_model = None
_gemini_available: Optional[bool] = None  # None = not yet tried

def _get_gemini():
    """Lazy-load Gemini generative model once."""
    global _gemini_model, _gemini_available
    if _gemini_available is not None:
        return _gemini_model

    # Read key from settings (pydantic-settings loads from .env automatically)
    try:
        from backend.app.core.config import settings
        api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
    except Exception:
        api_key = os.environ.get("GEMINI_API_KEY", "")

    if not api_key:
        logger.info("GEMINI_API_KEY not set — Gemini Vision OCR disabled. "
                    "Add GEMINI_API_KEY to .env for accurate handwriting recognition.")
        _gemini_available = False
        return None

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        _gemini_available = True
        logger.info("Gemini Vision model ready (gemini-2.5-flash).")
    except Exception as exc:
        logger.warning(f"Gemini not available: {exc}")
        _gemini_available = False

    return _gemini_model


def extract_all_fields_gemini(
    image_path: str,
    field_keys: list,
    template_key: str = "",
) -> Dict[str, Tuple[str, float]]:
    """
    Send the full form image to Gemini Flash Vision and extract ALL field values
    in a single API call.  Returns {field_key: (text, confidence)}.

    This is far more accurate than per-crop OCR because Gemini has full document
    context and understands label → value relationships natively.
    """
    model = _get_gemini()
    if model is None:
        return {}

    try:
        import google.generativeai as genai

        # Load image
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        # Build a prompt that lists the fields we want
        field_list = "\n".join(f"- {k}" for k in field_keys)
        prompt = f"""You are a highly accurate OCR system. Carefully read this scanned government form image and extract the handwritten or typed values for each of the following fields. 

Fields to extract:
{field_list}

Rules:
- Return ONLY the actual written value for each field (NOT the printed label text).
- For checkboxes: return "Checked" or "Unchecked".
- If a field is blank or illegible, return an empty string "".
- Return your answer as a valid JSON object with field names as keys and extracted text as string values.
- Do NOT include any explanation or markdown — just the raw JSON object.

Example format:
{{"full_name": "JOHN SMITH", "date_of_birth": "1990-01-15", "citizenship": "NAMIBIAN"}}
"""

        image_part = {"mime_type": "image/jpeg", "data": img_bytes}
        response = model.generate_content([prompt, image_part])
        raw = response.text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data = json.loads(raw)

        # Build return dict with confidence=0.92 for Gemini results
        result = {}
        for k in field_keys:
            val = str(data.get(k, "")).strip()
            result[k] = (val, 0.92)

        logger.info(f"Gemini extracted {len(result)} fields successfully.")
        return result

    except Exception as exc:
        logger.warning(f"Gemini form extraction failed: {exc}")
        return {}


# ─── Lazy-loaded global EasyOCR reader ────────────────────────────────────────
_easyocr_reader = None
_easyocr_available: Optional[bool] = None   # None = not yet tried

def _get_easyocr():
    """Load EasyOCR reader once, cache globally."""
    global _easyocr_reader, _easyocr_available
    if _easyocr_available is not None:
        return _easyocr_reader

    try:
        import easyocr
        logger.info("Loading EasyOCR reader (first call, may take ~20 s)…")
        _easyocr_reader = easyocr.Reader(
            ["en"],
            gpu=False,             # CPU-only for broad compatibility
            verbose=False,
            download_enabled=True  # auto-download model weights if missing
        )
        _easyocr_available = True
        logger.info("EasyOCR reader ready.")
    except Exception as exc:
        logger.warning(f"EasyOCR not available: {exc}. Will try pytesseract.")
        _easyocr_available = False

    return _easyocr_reader


# ─── Preprocessing helpers ─────────────────────────────────────────────────────

def _pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def remove_grid_lines(cv_img: np.ndarray) -> np.ndarray:
    """
    Removes horizontal and vertical ruled lines from a cropped field patch.
    
    Scanned forms have dense grid lines that EasyOCR misreads as pipes (|),
    dashes (-), or underscores (_). This function detects them with
    morphological operations and erases them by painting over with white.
    
    The approach:
      1. Binary threshold the grayscale image
      2. Detect long horizontal lines with a wide kernel
      3. Detect long vertical lines with a tall kernel  
      4. Dilate the detected lines slightly to cover anti-aliased edges
      5. Inpaint (paint white) over the detected line mask
    """
    if cv_img is None or cv_img.size == 0:
        return cv_img

    h, w = cv_img.shape[:2]
    if h < 10 or w < 10:
        return cv_img

    # Work on grayscale
    if len(cv_img.shape) == 3:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv_img.copy()

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect horizontal lines (kernel width = 30% of patch width, min 20px)
    h_size = max(20, w // 3)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)

    # Detect vertical lines (kernel height = 40% of patch height, min 15px)
    v_size = max(15, h // 3)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

    # Combine into a single mask and dilate slightly
    line_mask = cv2.add(h_lines, v_lines)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    line_mask = cv2.dilate(line_mask, dilate_kernel, iterations=1)

    # Paint over detected lines with white (255) on the original image
    result = cv_img.copy()
    if len(result.shape) == 3:
        result[line_mask > 0] = [255, 255, 255]
    else:
        result[line_mask > 0] = 255

    return result

def _cv2_to_pil(cv_img: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def preprocess_id_number(pil_img: Image.Image) -> Image.Image:
    """
    Specialized preprocessing for Namibian ID number fields.

    The ID is printed in a row of individual boxes separated by vertical
    ruled lines.  Standard grid-line removal erases horizontal lines but
    the VERTICAL separators cause EasyOCR to hallucinate '|', 'I', '1'
    between digits.  This function:
      1. Converts to grayscale and binarises with Otsu
      2. Detects & erases ALL vertical lines (any height >= 40 % of patch height)
      3. Detects & erases horizontal box borders as well
      4. Upscales 4x so each digit is large enough for confident recognition
      5. Applies CLAHE to boost ink contrast
    """
    img = _pil_to_cv2(pil_img)
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu binarisation on inverted image (ink = white in mask)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ── Erase vertical separator lines ────────────────────────────────────
    v_size = max(10, int(h * 0.4))   # lines at least 40 % of row height
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
    v_dilated = cv2.dilate(v_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)), iterations=2)

    # ── Erase horizontal box borders ───────────────────────────────────────
    h_size = max(10, int(w * 0.3))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
    h_dilated = cv2.dilate(h_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=1)

    # Combine line masks and paint white over them on the BGR image
    line_mask = cv2.add(v_dilated, h_dilated)
    cleaned = img.copy()
    cleaned[line_mask > 0] = [255, 255, 255]

    # ── Upscale 4x for small digit glyphs ─────────────────────────────────
    scale = max(4, 120 // max(1, h))
    cleaned = cv2.resize(cleaned, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # ── CLAHE on the upscaled image ────────────────────────────────────────
    lab = cv2.cvtColor(cleaned, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    cleaned = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return _cv2_to_pil(cleaned)

def preprocess_printed(pil_img: Image.Image) -> Image.Image:
    """
    Preprocessing for printed/mixed form fields.
    
    Pipeline: grid line removal → 2x upscale → CLAHE on LAB → mild denoise.
    
    Key insight: EasyOCR's CRAFT+CRNN pipeline works best on natural color/grayscale.
    Binarization destroys ink gradients that the neural network relies on.
    Grid line removal is done BEFORE upscaling to avoid amplifying line artifacts.
    """
    img = _pil_to_cv2(pil_img)

    # Step 0: Remove form grid lines (prevents OCR misreading lines as |, -, _)
    img = remove_grid_lines(img)

    h, w = img.shape[:2]

    # Always upscale to at least 2x — gives EasyOCR much more detail to work with
    target_h = max(h * 2, 80)
    target_w = max(w * 2, 160)
    scale = max(target_h / h, target_w / w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # CLAHE on LAB color space — enhances local contrast without blowing out ink
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Mild color denoising — removes scanner noise without smearing ink strokes
    img = cv2.fastNlMeansDenoisingColored(img, h=6, hColor=6)

    return _cv2_to_pil(img)


def preprocess_handwriting(pil_img: Image.Image) -> Image.Image:
    """
    Preprocessing optimized for handwritten field patches.
    
    Handwriting OCR benefits from: strong upscaling (3x) + bilateral edge-preserving
    filter + CLAHE. We still avoid hard binarization because EasyOCR handles the
    thresholding internally in its detection/recognition pipeline.
    """
    img = _pil_to_cv2(pil_img)
    h, w = img.shape[:2]

    # Upscale 3x for handwriting — thin ink strokes need more pixels
    target_h = max(h * 3, 120)
    target_w = max(w * 3, 240)
    scale = max(target_h / h, target_w / w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # Strong CLAHE on LAB — boosts faint handwriting without destroying ink color
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Bilateral filter — edge-preserving smoothing keeps ink sharp, removes noise
    img = cv2.bilateralFilter(img, 9, 75, 75)

    return _cv2_to_pil(img)


def check_checkbox(pil_img: Image.Image) -> Tuple[str, float]:
    """Detects whether a checkbox region is checked using pixel density."""
    arr = np.array(pil_img.convert("L"))
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    density = np.sum(binary == 255) / binary.size
    if density > 0.18:
        return "Checked", 0.95
    elif density > 0.08:
        return "Partially marked", 0.70
    return "Unchecked", 0.92


# ─── Core OCR runners ─────────────────────────────────────────────────────────

def _run_easyocr(
    pil_img: Image.Image,
    allowlist: Optional[str] = None,
    low_text: float = 0.3,
) -> Tuple[str, float]:
    """Run EasyOCR on a PIL image patch. Returns (text, confidence).

    Args:
        pil_img:   Pre-processed PIL image.
        allowlist: If set, restrict recognised characters to this string
                   (e.g. '0123456789' for numeric-only fields).
        low_text:  EasyOCR low_text threshold (lower = detect faint text).
    """
    reader = _get_easyocr()
    if reader is None:
        raise RuntimeError("EasyOCR reader not available.")

    arr = np.array(pil_img.convert("RGB"))

    kwargs = dict(detail=1, paragraph=False, low_text=low_text)
    if allowlist:
        kwargs["allowlist"] = allowlist

    # EasyOCR returns list of (bbox, text, prob)
    results = reader.readtext(arr, **kwargs)

    if not results:
        return "", 0.0

    texts = []
    confidences = []
    for (_bbox, text, prob) in results:
        t = text.strip()
        if t:
            texts.append(t)
            confidences.append(prob)

    combined_text = " ".join(texts)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return combined_text, round(avg_conf, 3)


def _run_tesseract(pil_img: Image.Image, psm: int = 7) -> Tuple[str, float]:
    """Run pytesseract on a PIL image patch. Returns (text, confidence)."""
    import pytesseract

    # Check for configured path
    tess_cmd = os.environ.get("TESSERACT_CMD", "")
    if tess_cmd and os.path.exists(tess_cmd):
        pytesseract.pytesseract.tesseract_cmd = tess_cmd

    config = f"--psm {psm} --oem 1"
    data = pytesseract.image_to_data(pil_img, config=config, output_type=pytesseract.Output.DICT)

    words, confs = [], []
    for i, word in enumerate(data["text"]):
        w = word.strip()
        c = float(data["conf"][i])
        if w and c >= 0:
            words.append(w)
            confs.append(c)

    text = " ".join(words)
    avg_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
    return text, round(avg_conf, 3)


# ─── Public API ──────────────────────────────────────────────────────────────

# Fields that contain only numeric/alphanumeric box-separated digits
_ID_FIELDS = {"namibian_id", "date_of_birth"}
# Fields where we expect only alphanumeric characters (no special symbols)
_ALPHA_FIELDS = {"citizenship", "full_name", "position_applied", "duty_station",
                 "office_ministry", "assume_duty", "postal_address",
                 "residential_address", "phone_number", "email_address"}


def extract_field_text(
    pil_img: Image.Image,
    field_type: str = "printed_handwritten",
    field_key: str = "",
    mock_fallback: Optional[Tuple[str, float]] = None
) -> Tuple[str, float, str]:
    """
    Extract text from a cropped field image.

    Args:
        pil_img:       Cropped PIL image of the field region.
        field_type:    "printed_handwritten" | "handwriting" | "checkbox"
        field_key:     Field name (for logging).
        mock_fallback: (text, confidence) tuple to use if all real engines fail.

    Returns:
        (text, confidence, engine_name)
    """
    # ── Checkboxes: no OCR needed ──────────────────────────────────────────
    if field_type == "checkbox":
        text, conf = check_checkbox(pil_img)
        return text, conf, "opencv_density"

    # ── ID number fields: use specialised digit preprocessor ───────────────
    is_id_field = field_key in _ID_FIELDS
    is_handwriting = (field_type == "handwriting")

    if is_id_field:
        try:
            preprocessed = preprocess_id_number(pil_img)
            # Restrict EasyOCR to digits + slash + hyphen for ID/DOB fields
            allowlist = "0123456789/-"
            text, conf = _run_easyocr(preprocessed, allowlist=allowlist, low_text=0.2)
            if not text.strip():
                # Fallback: try without allowlist in case the field has letters (e.g. old ID)
                text, conf = _run_easyocr(preprocessed, low_text=0.2)
            if text.strip():
                # Clean up spaces between digits (boxes cause them)
                import re
                clean = re.sub(r"\s+", "", text.strip())
                logger.debug(f"EasyOCR ID [{field_key}]: '{clean}' ({conf:.2f})")
                return clean, conf, "easyocr_id"
        except Exception as e:
            logger.debug(f"ID preprocessing failed for '{field_key}': {e}")

    # ── Preprocess based on type ──────────────────────────────────────────
    try:
        preprocessed = preprocess_handwriting(pil_img) if is_handwriting else preprocess_printed(pil_img)
    except Exception as e:
        logger.warning(f"Preprocessing failed for '{field_key}': {e}. Using raw patch.")
        preprocessed = pil_img

    # ── Try EasyOCR first ──────────────────────────────────────────────────
    try:
        text, conf = _run_easyocr(preprocessed, low_text=0.25)
        if text.strip():
            logger.debug(f"EasyOCR [{field_key}]: '{text}' ({conf:.2f})")
            return text.strip(), conf, "easyocr"
        # No text detected — try on original (sometimes preprocessing is too aggressive)
        text, conf = _run_easyocr(pil_img, low_text=0.25)
        if text.strip():
            return text.strip(), conf, "easyocr_raw"
    except Exception as e:
        logger.debug(f"EasyOCR failed for '{field_key}': {e}")

    # ── Try pytesseract fallback ───────────────────────────────────────────
    try:
        psm = 8 if is_handwriting else 7   # 8=single word, 7=single line
        text, conf = _run_tesseract(preprocessed, psm=psm)
        if text.strip():
            return text.strip(), conf, "tesseract"
    except Exception as e:
        logger.debug(f"pytesseract failed for '{field_key}': {e}")

    # ── Mock fallback (demo safety net) ───────────────────────────────────
    if mock_fallback:
        text, conf = mock_fallback
        logger.info(f"Using mock fallback for '{field_key}': '{text}'")
        return text, conf, "mock_fallback"

    return "", 0.0, "no_engine"
