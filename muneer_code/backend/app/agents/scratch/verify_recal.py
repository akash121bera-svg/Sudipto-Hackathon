"""
Verify recalibrated bboxes — saves overlay + individual crops for all 12 fields.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

import cv2
import numpy as np

UPLOAD_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "storage", "uploads"))
test_image = next((os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))), None)

img = cv2.imread(test_image)
h, w = img.shape[:2]
print(f"Image: {w}x{h}")

# Recalibrated normalized bboxes (0-1000 scale) → convert to pixels
norm_fields = {
    "position_applied":    [225,  13, 279, 479],
    "office_ministry":     [225, 482, 279, 990],
    "duty_station":        [283,  13, 333, 479],
    "assume_duty":         [254, 482, 333, 990],
    "full_name":           [439, 234, 484, 723],
    "namibian_id":         [486,  13, 529, 352],
    "date_of_birth":       [486, 495, 529, 723],
    "citizenship":         [551,  13, 591, 352],
    "postal_address":      [596,  13, 666, 352],
    "residential_address": [596, 495, 666, 723],
    "phone_number":        [670,  13, 709, 260],
    "email_address":       [686,  13, 729, 723],
}

def to_px(ymin, xmin, ymax, xmax):
    return (int(ymin*h/1000), int(xmin*w/1000), int(ymax*h/1000), int(xmax*w/1000))

overlay = img.copy()
colors = [(0,200,0), (255,80,0), (0,80,255), (200,200,0), (0,200,200), (200,0,200),
          (0,160,0), (180,60,0), (0,60,200), (160,160,0), (0,160,160), (160,0,160)]

crops_dir = os.path.join(os.path.dirname(__file__), "recal_crops")
os.makedirs(crops_dir, exist_ok=True)

for i, (key, norm) in enumerate(norm_fields.items()):
    ymin, xmin, ymax, xmax = to_px(*norm)
    color = colors[i % len(colors)]
    cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.putText(overlay, key[:14], (xmin+2, ymin+14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    
    crop = img[ymin:ymax, xmin:xmax]
    scale = max(1, min(4, 120 // max(1, ymax - ymin)))
    crop_big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(crops_dir, f"{key}.png"), crop_big)
    print(f"  {key:25s}: px=[y{ymin}-{ymax}, x{xmin}-{xmax}]  size={xmax-xmin}w x {ymax-ymin}h")

out = os.path.join(os.path.dirname(__file__), "recal_overlay.png")
cv2.imwrite(out, overlay)
print(f"\nSaved overlay: {out}")
print(f"Crops: {crops_dir}")
