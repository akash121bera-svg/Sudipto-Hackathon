"""
Visual survey: draw horizontal lines at key y positions to find where each form row starts.
Shows a full-width 1px slice annotation every 20px from y=150 to y=700.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
import cv2
import numpy as np

UPLOAD_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "storage", "uploads"))
test_image = next((os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))), None)

img = cv2.imread(test_image)
h, w = img.shape[:2]

# Draw a line every 20 pixels with the y label
survey = img.copy()
for y in range(150, min(700, h), 10):
    color = (0, 0, 255) if y % 50 == 0 else (0, 180, 0)
    thickness = 2 if y % 50 == 0 else 1
    cv2.line(survey, (0, y), (w, y), color, thickness)
    if y % 20 == 0:
        cv2.putText(survey, f"y={y}", (w-70, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

out = os.path.join(os.path.dirname(__file__), "row_survey.png")
cv2.imwrite(out, survey)
print(f"Saved: {out}")

# Also save a zoomed lower section (y=400-700) where most personal fields are
zone = img[400:700, :].copy()
for y in range(0, 300, 10):
    abs_y = y + 400
    color = (0, 0, 255) if abs_y % 50 == 0 else (0, 180, 0)
    thickness = 2 if abs_y % 50 == 0 else 1
    cv2.line(zone, (0, y), (w, y), color, thickness)
    if abs_y % 20 == 0:
        cv2.putText(zone, f"y={abs_y}", (w-70, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

# Scale 2x for readability
zone_big = cv2.resize(zone, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
out2 = os.path.join(os.path.dirname(__file__), "row_survey_lower.png")
cv2.imwrite(out2, zone_big)
print(f"Saved: {out2}")
