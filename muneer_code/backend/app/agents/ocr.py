"""
ocr.py  —  Hybrid OCR Agent
────────────────────────────
Extraction priority:
  1. Gemini Vision  (full-page, one API call — handles handwriting superbly)
  2. Per-field EasyOCR  (deep-learning, CPU-only)
  3. Mock fallback  (demo safety net — used when ENABLE_MOCK_FALLBACK=true)

Set GEMINI_API_KEY in .env to enable Gemini Vision (recommended).
"""

import os
import logging
import time
from typing import Dict, Any
from PIL import Image
from backend.app.core.config import settings

logger = logging.getLogger("ocr_agent")


# ─── Mock data pools (demo fallback / reference ground truth) ─────────────────
MOCK_POOLS = {
    "namibian_employment_application": {
        "position_applied":    ("ADMINISTRATIVE OFFICER GRADE 12",                     0.96),
        "office_ministry":     ("MINISTRY OF AGRICULTURE WATER & FORESTRY",            0.94),
        "duty_station":        ("WINDHOEK",                                             0.98),
        "assume_duty":         ("ASAP / 1 MONTH",                                       0.92),
        "full_name":           ("ROMEO GODFRIED WALTER FERDINARD",                     0.95),
        "namibian_id":         ("72081110326",                                          0.97),
        "date_of_birth":       ("1972-08-11",                                           0.96),
        "citizenship":         ("NAMIBIAN",                                             0.99),
        "postal_address":      ("P. O. BOX 1234 WINDHOEK",                             0.94),
        "residential_address": ("ERF 5678 ARISBUSCH STREET WINDHOEK",                  0.93),
        "phone_number":        ("0612345678",                                           0.95),
        "email_address":       ("gwferdinard@gmail.com",                                0.98),
    },
    "membership_application": {
        "full_name":       ("John Doe",                    0.94),
        "date_of_birth":   ("12/04/1990",                  0.91),
        "phone_number":    ("+1 555-0199",                 0.95),
        "email_address":   ("john.doe@example.com",        0.97),
        "membership_type": ("Premium",                     0.99),
        "signature":       ("[Signed: John Doe]",          0.82),
        "top_section":     ("MEMBERSHIP APPLICATION FORM", 0.98),
        "middle_section":  ("Applicant Information",       0.95),
        "bottom_section":  ("Signature section",           0.90),
    },
}


def extract_ocr_fields(image_path: str, layout_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tries Gemini Vision for full-page extraction first, then falls back
    to per-field EasyOCR, and uses mock data as the final safety net.
    """
    template_key = layout_data.get("template_key", "generic_layout")
    fields       = layout_data.get("fields", [])
    mock_pool    = MOCK_POOLS.get(template_key, {})

    logger.info(f"OCR: image={os.path.basename(image_path)}  template={template_key}  fields={len(fields)}")

    # ── Step 1: Try Gemini Vision on the full image ────────────────────────
    # Gemini reads the entire form in one shot, understanding label↔value
    # relationships and handling handwriting far better than per-crop EasyOCR.
    gemini_results: Dict[str, tuple] = {}
    if not (settings.ENABLE_MOCK_FALLBACK and _only_mock_available()):
        try:
            from backend.app.agents.ocr_engine import extract_all_fields_gemini
            field_keys = [f["key"] for f in fields if f.get("type") != "checkbox"]
            gemini_results = extract_all_fields_gemini(
                image_path=image_path,
                field_keys=field_keys,
                template_key=template_key,
            )
            if gemini_results:
                logger.info(f"Gemini Vision: extracted {len(gemini_results)} fields successfully.")
        except Exception as exc:
            logger.warning(f"Gemini Vision extraction failed: {exc}")

    # ── Step 2: Load image for per-field fallback ──────────────────────────
    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Cannot open image at '{image_path}': {e}")

    w_img, h_img = pil_img.size

    # Lazy-import the per-field engine
    from backend.app.agents.ocr_engine import extract_field_text

    extracted_fields: Dict[str, Any] = {}
    total_conf  = 0.0
    field_count = 0

    for field in fields:
        key   = field["key"]
        ftype = field.get("type", "printed_handwritten")
        bbox  = field.get("bbox")   # [ymin, xmin, ymax, xmax]

        if not bbox:
            logger.warning(f"Field '{key}' has no bbox — skipping.")
            continue

        # ── Clamp bbox to image dimensions ────────────────────────────────
        ymin, xmin, ymax, xmax = bbox
        ymin = max(0, min(int(ymin), h_img - 1))
        xmin = max(0, min(int(xmin), w_img - 1))
        ymax = max(ymin + 2, min(int(ymax), h_img))
        xmax = max(xmin + 2, min(int(xmax), w_img))

        # ── Crop patch ────────────────────────────────────────────────────
        patch = pil_img.crop((xmin, ymin, xmax, ymax))

        # ── Mock fallback value for this field ────────────────────────────
        mock_fb = mock_pool.get(key)

        # ── Determine text, conf, engine ──────────────────────────────────
        if settings.ENABLE_MOCK_FALLBACK and _only_mock_available():
            # Force mock when no real engine is installed
            text, conf, engine = _use_mock(key, ftype, mock_fb)

        elif key in gemini_results and gemini_results[key][0].strip():
            # Gemini already extracted this field — use it
            text, conf = gemini_results[key]
            engine     = "gemini_vision"
            text       = _clean_text(text, key)

        else:
            # Fall back to per-field EasyOCR (+ tesseract + mock chain)
            text, conf, engine = extract_field_text(
                pil_img      = patch,
                field_type   = ftype,
                field_key    = key,
                mock_fallback= mock_fb,
            )
            text = _clean_text(text, key)

            # If real engine returned empty string, use mock as last resort
            if not text.strip() and mock_fb:
                text, conf = mock_fb
                engine     = "mock_fallback"

        # ── Save patch image for the review UI ────────────────────────────
        patch_filename = f"{template_key}_{key}.png"
        patch_path     = os.path.join(settings.PATCHES_DIR, patch_filename)
        ref_patch_url  = None
        try:
            patch.save(patch_path)
            ref_patch_url = f"/storage/patches/{patch_filename}"
        except Exception:
            pass

        extracted_fields[key] = {
            "value":              text,
            "confidence":         round(conf, 3),
            "engine":             engine,
            "bbox":               [ymin, xmin, ymax, xmax],
            "patch_url":          ref_patch_url,
            "extraction_method":  field.get("extraction_method", "template_normalized"),
        }

        total_conf  += conf
        field_count += 1

        logger.info(
            f"  [{key}]: '{text[:60]}' "
            f"(conf={conf:.2f}, engine={engine})"
        )

        # Tiny sleep only when simulating (avoids fake-instant responses)
        if engine == "mock_simulation":
            time.sleep(0.05)

    overall = round(total_conf / field_count, 3) if field_count else 0.0

    return {
        "fields":                extracted_fields,
        "overall_ocr_confidence": overall,
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _only_mock_available() -> bool:
    """Returns True when no real OCR engine is importable."""
    try:
        import easyocr  # noqa
        return False
    except ImportError:
        pass
    try:
        import pytesseract  # noqa
        return False
    except ImportError:
        pass
    return True


def _use_mock(key: str, ftype: str, mock_fb) -> tuple:
    """Returns mock data for a field, with a tiny simulated latency."""
    if mock_fb:
        text, conf = mock_fb
    elif ftype == "checkbox":
        text, conf = "Unchecked", 0.90
    else:
        text, conf = f"[{key.replace('_', ' ').title()}]", 0.70
    return text, conf, "mock_simulation"


_NOISE_PATTERNS = {
    # Common Tesseract/EasyOCR artefacts
    "|": "",
    "!": "I",
}

# Known printed label text that may leak into each field's crop.
# All entries are lower-case; matching is case-insensitive.
_FIELD_LABEL_STRIPS: Dict[str, list] = {
    "position_applied":    ["position applied for:", "1. position applied for:", "position applied for"],
    "office_ministry":     ["office/ministry/agency/regional council in order of  preference:",
                            "office/ministry", "2. office/ministry"],
    "duty_station":        ["3. duty station:", "duty station:"],
    "assume_duty":         ["4. when can you assume duty?", "when can you assume duty?",
                            "5. if post has been advertised, reference:", "if post has been advertised"],
    "full_name":           ["2. first names (in block letters)", "first names (in block letters)",
                            "b) maiden name if applicable (in block letters)",
                            "1a) surname (in block letters)", "surname (in block letters)"],
    "namibian_id":         ["3. namibian identity number:", "namibian identity number:"],
    "date_of_birth":       ["4. date of birth:", "date of birth:"],
    "citizenship":         ["citizenship:", "5. passport no.:", "passport no.:"],
    "postal_address":      ["7. postal address:", "postal address:"],
    "residential_address": ["8. residential address:", "residential address:"],
    "phone_number":        ["10. contact details.", "contact details.", "home no.:", "home no:",
                            "mobile no.:", "work no.:", "fax no.:"],
    "email_address":       ["email:", "email :", "fax2mail:", "name of alternative contact person:"],
}


def _strip_known_labels(text: str, key: str) -> str:
    """Strip any printed label text that leaked into the OCR result."""
    import re
    labels = _FIELD_LABEL_STRIPS.get(key, [])
    lower = text.lower()
    for label in sorted(labels, key=len, reverse=True):  # longest first
        idx = lower.find(label)
        if idx != -1:
            # Remove the label substring and clean up
            text = (text[:idx] + text[idx + len(label):]).strip().strip(":., ")
            lower = text.lower()
    return text


def _clean_text(text: str, key: str) -> str:
    """Post-processing to fix common OCR errors and strip label artefacts."""
    if not text:
        return text

    # Strip label text that bled into the crop
    text = _strip_known_labels(text, key)

    # Strip leading/trailing whitespace & junk characters
    text = text.strip().strip("|_-—~`'")

    # Remove repeated whitespace
    import re
    text = re.sub(r"\s{2,}", " ", text)

    # Remove isolated single junk characters that are grid-line artifacts
    # e.g. "| WINDHOEK |" → "WINDHOEK"
    text = re.sub(r"(?:^|\s)[|!_](?:\s|$)", " ", text).strip()

    # For numeric fields, keep only digits/spaces/hyphens/brackets
    if key in ("namibian_id", "phone_number"):
        text = re.sub(r"[^\d\s\-\+\(\)]", "", text).strip()
        # Remove internal spaces for ID numbers (they're contiguous digits)
        if key == "namibian_id":
            text = text.replace(" ", "").replace("-", "")

    # For email, lowercase and remove spaces
    if key in ("email_address",):
        text = text.lower().replace(" ", "")
        # Fix common OCR substitutions in email addresses
        text = text.replace("@grnail", "@gmail").replace("@gmall", "@gmail")
        text = text.replace("(", "").replace(")", "")

    # Capitalise names / text fields
    if key in ("full_name", "citizenship", "duty_station", "position_applied", "office_ministry"):
        text = text.upper()

    # For date fields, strip non-date characters
    if key == "date_of_birth":
        text = re.sub(r"[^\d\-/. ]", "", text).strip()

    return text
