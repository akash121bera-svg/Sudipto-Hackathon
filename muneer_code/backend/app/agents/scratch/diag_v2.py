"""
Visual diagnostic: draw current bboxes on the image and save individual crops.
Run: .\\venv\\Scripts\\python.exe backend\\app\\agents\\scratch\\diag_v2.py
"""
import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

UPLOAD_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "storage", "uploads"))
OUT_DIR = os.path.join(os.path.dirname(__file__), "diag_v2_crops")
os.makedirs(OUT_DIR, exist_ok=True)

img_file = next((os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))), None)
if not img_file:
    print("No image found!"); sys.exit(1)

img = cv2.imread(img_file)
h, w = img.shape[:2]
print(f"Image: {os.path.basename(img_file)}  dims={w}x{h}")

# Current template bboxes [ymin, xmin, ymax, xmax] in 0-1000 normalized
fields = [
    {"key": "position_applied",    "bbox": [240,  13, 270, 490], "color": (0,   200,   0)},
    {"key": "office_ministry",     "bbox": [240, 495, 295, 990], "color": (0,   150,   0)},
    {"key": "duty_station",        "bbox": [303,  13, 338, 490], "color": (255,   0,   0)},
    {"key": "assume_duty",         "bbox": [303, 495, 338, 990], "color": (200,   0,   0)},
    {"key": "full_name",           "bbox": [429, 234, 467, 723], "color": (0,     0, 255)},
    {"key": "namibian_id",         "bbox": [467,  13, 503, 352], "color": (0,   100, 200)},
    {"key": "date_of_birth",       "bbox": [467, 430, 503, 723], "color": (0,    50, 200)},
    {"key": "citizenship",         "bbox": [515, 185, 538, 352], "color": (150, 150,   0)},
    {"key": "postal_address",      "bbox": [540,  13, 590, 352], "color": (200, 100,   0)},
    {"key": "residential_address", "bbox": [540, 430, 590, 723], "color": (200,  50,   0)},
    {"key": "phone_number",        "bbox": [618, 145, 640, 330], "color": (150,   0, 200)},
    {"key": "email_address",       "bbox": [638,  90, 660, 660], "color": (100,   0, 200)},
]

overlay = img.copy()
font = cv2.FONT_HERSHEY_SIMPLEX

for f in fields:
    yn, xn, yx, xx = f["bbox"]
    y0 = int(yn * h / 1000)
    x0 = int(xn * w / 1000)
    y1 = int(yx * h / 1000)
    x1 = int(xx * w / 1000)

    print(f"  {f['key']:<25} norm={yn},{xn},{yx},{xx}  px=({x0},{y0})-({x1},{y1})  size={x1-x0}x{y1-y0}")

    cv2.rectangle(overlay, (x0, y0), (x1, y1), f["color"], 2)
    cv2.putText(overlay, f["key"][:15], (x0 + 2, y0 + 12), font, 0.35, f["color"], 1)

    # Save crop
    crop = img[y0:y1, x0:x1]
    if crop.size > 0:
        cv2.imwrite(os.path.join(OUT_DIR, f"{f['key']}.png"), crop)

cv2.imwrite(os.path.join(OUT_DIR, "overlay.png"), overlay)
print(f"\nOverlay saved: {OUT_DIR}\\overlay.png")
print(f"Crops saved:  {OUT_DIR}\\")
