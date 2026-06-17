"""
Diagnostic: Save all field crops with labels to understand exact bbox positions.
Also shows the full image with bbox overlays.
Run this to calibrate the Namibian form template.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "storage", "uploads")
UPLOAD_DIR = os.path.normpath(UPLOAD_DIR)

test_image = None
for fn in os.listdir(UPLOAD_DIR):
    if fn.endswith((".jpg", ".jpeg", ".png")):
        test_image = os.path.join(UPLOAD_DIR, fn)
        break

if not test_image:
    print("No test image found!")
    sys.exit(1)

img = cv2.imread(test_image)
h, w = img.shape[:2]
print(f"Image size: {w}x{h}")

# Known fields from layout output:
# bbox = [ymin, xmin, ymax, xmax]
fields = {
    "position_applied":    [177, 168, 231, 368],
    "office_ministry":     [177, 430, 231, 675],
    "duty_station":        [231, 168, 275, 368],
    "assume_duty":         [231, 430, 275, 675],
    "full_name":           [328, 192, 373, 552],
    "namibian_id":         [373, 168, 412, 330],
    "date_of_birth":       [373, 414, 412, 552],
    "citizenship":         [418, 168, 467, 330],
    "postal_address":      [473, 168, 506, 330],
    "residential_address": [473, 414, 506, 552],
    "phone_number":        [515, 168, 551, 330],
    "email_address":       [530, 168, 577, 330],
}

# Draw overlays on full image
overlay = img.copy()
colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)]
for i, (key, (ymin, xmin, ymax, xmax)) in enumerate(fields.items()):
    color = colors[i % len(colors)]
    cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.putText(overlay, key[:12], (xmin, max(ymin-3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

out_path = os.path.join(os.path.dirname(__file__), "field_overlay_diag.png")
cv2.imwrite(out_path, overlay)
print(f"Saved overlay: {out_path}")

# Save each crop
crops_dir = os.path.join(os.path.dirname(__file__), "diag_crops")
os.makedirs(crops_dir, exist_ok=True)

for key, (ymin, xmin, ymax, xmax) in fields.items():
    crop = img[ymin:ymax, xmin:xmax]
    # Upscale 3x for visibility
    crop_big = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    out = os.path.join(crops_dir, f"{key}.png")
    cv2.imwrite(out, crop_big)
    print(f"  {key:25s}: size={xmax-xmin}x{ymax-ymin}  -> {out}")

print("\nDone. Check diag_crops/ folder and field_overlay_diag.png")
