"""
Targeted EasyOCR diagnostic — tests different preprocessing pipelines on each crop.
Run: .\\venv\\Scripts\\python.exe backend\\app\\agents\\scratch\\easyocr_tune.py
"""
import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

import easyocr
reader = easyocr.Reader(["en"], gpu=False, verbose=False)

UPLOAD_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "storage", "uploads"))
img_file = next((os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))), None)
if not img_file:
    print("No image found!"); sys.exit(1)

img = cv2.imread(img_file)
h, w = img.shape[:2]
print(f"Image: {os.path.basename(img_file)}  {w}x{h}\n")

# Test fields with EXPANDED bboxes (more vertical room)
fields = [
    # key, tight_bbox, expanded_bbox
    ("position_applied", [240, 13, 270, 490],  [200, 13, 285, 490]),
    ("office_ministry",  [240, 495, 295, 990], [200, 495, 310, 990]),
    ("duty_station",     [303, 13, 338, 490],  [290, 13, 355, 490]),
    ("full_name",        [429, 234, 467, 723],  [420, 234, 475, 723]),
]

def to_px(yn, xn, yx, xx):
    return int(yn*h/1000), int(xn*w/1000), int(yx*h/1000), int(xx*w/1000)

def try_ocr(arr, low_text=0.3):
    """Run EasyOCR with given settings and return text."""
    results = reader.readtext(arr, detail=1, paragraph=False, low_text=low_text)
    if not results:
        return ""
    texts = [t.strip() for _, t, _ in results if t.strip()]
    return " ".join(texts)

def upscale(crop_cv, scale=4):
    return cv2.resize(crop_cv, (crop_cv.shape[1]*scale, crop_cv.shape[0]*scale), interpolation=cv2.INTER_CUBIC)

def clahe_enhance(crop_cv):
    lab = cv2.cvtColor(crop_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    lab = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def binarize(crop_cv):
    gray = cv2.cvtColor(crop_cv, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

for key, tight, expanded in fields:
    print(f"\n{'='*70}")

    for label, bbox in [("TIGHT", tight), ("EXPANDED", expanded)]:
        y0, x0, y1, x1 = to_px(*bbox)
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            print(f"[{key}:{label}] EMPTY"); continue

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        up4 = upscale(crop, 4)
        up4_rgb = cv2.cvtColor(up4, cv2.COLOR_BGR2RGB)
        up4_clahe = clahe_enhance(up4)
        up4_clahe_rgb = cv2.cvtColor(up4_clahe, cv2.COLOR_BGR2RGB)
        up4_bw = binarize(up4)
        up4_bw_rgb = cv2.cvtColor(up4_bw, cv2.COLOR_BGR2RGB)

        print(f"\nField: {key} [{label}]  size={x1-x0}x{y1-y0}px")
        print(f"  raw             : {repr(try_ocr(rgb)[:80])}")
        print(f"  4x              : {repr(try_ocr(up4_rgb)[:80])}")
        print(f"  4x+CLAHE        : {repr(try_ocr(up4_clahe_rgb)[:80])}")
        print(f"  4x+binarize     : {repr(try_ocr(up4_bw_rgb)[:80])}")
        print(f"  4x+lt=0.15      : {repr(try_ocr(up4_clahe_rgb, low_text=0.15)[:80])}")

print("\nDone.")
