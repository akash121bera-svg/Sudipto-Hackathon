"""
E2E OCR test — runs full field extraction pipeline on the uploaded form image.
Shows extracted text, confidence, and engine per field.
"""
import sys, os, logging
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

logging.basicConfig(level=logging.WARNING)

import cv2
from PIL import Image
from backend.app.agents.ocr_engine import extract_field_text, preprocess_id_number, preprocess_printed

UPLOAD_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "storage", "uploads"))
test_image = next((os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))), None)
if not test_image:
    print("No image found!"); sys.exit(1)

print(f"Testing on: {os.path.basename(test_image)}\n")

img_cv = cv2.imread(test_image)
h, w = img_cv.shape[:2]

# Final calibrated bboxes [ymin, xmin, ymax, xmax] normalized 0-1000
fields = [
    {"key": "position_applied",    "type": "printed_handwritten", "bbox": [240,  13, 270, 490]},
    {"key": "office_ministry",     "type": "printed_handwritten", "bbox": [240, 495, 295, 990]},
    {"key": "duty_station",        "type": "printed_handwritten", "bbox": [303,  13, 338, 490]},
    {"key": "assume_duty",         "type": "printed_handwritten", "bbox": [303, 495, 338, 990]},
    {"key": "full_name",           "type": "printed_handwritten", "bbox": [429, 234, 467, 723]},
    {"key": "namibian_id",         "type": "printed_handwritten", "bbox": [467,  13, 503, 352]},
    {"key": "date_of_birth",       "type": "printed_handwritten", "bbox": [467, 430, 503, 723]},
    {"key": "citizenship",         "type": "printed_handwritten", "bbox": [515, 185, 538, 352]},
    {"key": "postal_address",      "type": "printed_handwritten", "bbox": [540,  13, 590, 352]},
    {"key": "residential_address", "type": "printed_handwritten", "bbox": [540, 430, 590, 723]},
    {"key": "phone_number",        "type": "printed_handwritten", "bbox": [618, 145, 640, 330]},
    {"key": "email_address",       "type": "printed_handwritten", "bbox": [638,  90, 660, 660]},
]

def to_px(ymin, xmin, ymax, xmax):
    return int(ymin*h/1000), int(xmin*w/1000), int(ymax*h/1000), int(xmax*w/1000)

print(f"{'FIELD':<25} {'RAW TEXT':<40} {'CLEANED':<30} {'CONF':>6}  ENGINE")
print("-" * 110)

from backend.app.agents.ocr import _clean_text

for f in fields:
    ymin, xmin, ymax, xmax = to_px(*f["bbox"])
    crop_cv = img_cv[ymin:ymax, xmin:xmax]
    if crop_cv.size == 0:
        print(f"  {f['key']:<23} [EMPTY CROP]")
        continue
    pil_crop = Image.fromarray(cv2.cvtColor(crop_cv, cv2.COLOR_BGR2RGB))
    text, conf, engine = extract_field_text(pil_crop, field_type=f["type"], field_key=f["key"])
    cleaned = _clean_text(text, f["key"])
    conf_pct = f"{conf*100:.0f}%"
    print(f"  {f['key']:<23} {repr(text):<40} {repr(cleaned):<30} {conf_pct:>6}  [{engine}]")

print("\nDone.")
