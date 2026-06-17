"""
Verify final recalibrated bboxes — saves overlay + individual crops for all 12 fields.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

import cv2

UPLOAD_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "storage", "uploads"))
test_image = next((os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))), None)

img = cv2.imread(test_image)
h, w = img.shape[:2]
print(f"Image: {w}x{h}")

# Final calibrated normalized bboxes (from row survey)
norm_fields = {
    "position_applied":    [225,  13, 270, 490],
    "office_ministry":     [225, 495, 295, 990],
    "duty_station":        [290,  13, 338, 490],
    "assume_duty":         [290, 495, 338, 990],
    "full_name":           [429, 234, 467, 723],
    "namibian_id":         [467,  13, 503, 352],
    "date_of_birth":       [467, 430, 503, 723],
    "citizenship":         [515,  13, 550, 352],
    "postal_address":      [540,  13, 590, 352],
    "residential_address": [540, 430, 590, 723],
    "phone_number":        [607,  13, 640, 320],
    "email_address":       [628,  13, 660, 723],
}

def to_px(ymin, xmin, ymax, xmax):
    return (int(ymin*h/1000), int(xmin*w/1000), int(ymax*h/1000), int(xmax*w/1000))

overlay = img.copy()
colors = [(0,200,0), (255,80,0), (0,80,255), (200,200,0), (0,200,200), (200,0,200),
          (0,160,0), (180,60,0), (0,60,200), (160,160,0), (0,160,160), (160,0,160)]

crops_dir = os.path.join(os.path.dirname(__file__), "final_crops")
os.makedirs(crops_dir, exist_ok=True)

for i, (key, norm) in enumerate(norm_fields.items()):
    ymin, xmin, ymax, xmax = to_px(*norm)
    color = colors[i % len(colors)]
    cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.putText(overlay, key[:14], (xmin+2, ymin+14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    crop = img[ymin:ymax, xmin:xmax]
    scale = max(2, min(5, 120 // max(1, ymax - ymin)))
    crop_big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(crops_dir, f"{key}.png"), crop_big)
    print(f"  {key:25s}: px=[y{ymin}-{ymax}, x{xmin}-{xmax}] size={xmax-xmin}w x {ymax-ymin}h")

out = os.path.join(os.path.dirname(__file__), "final_overlay.png")
cv2.imwrite(out, overlay)
print(f"\nOverlay: {out}")
print(f"Crops:   {crops_dir}")
