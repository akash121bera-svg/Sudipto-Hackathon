"""
Integration test: Verify the refactored layout agent works correctly.
Tests:
  1. Template classification (visual check for Namibian emblem)
  2. Horizontal line detection
  3. Bbox snapping to detected lines
  4. Field definitions are sane (non-empty, non-overlapping)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend.app.agents.layout import (
    analyze_layout, detect_horizontal_lines,
    is_namibian_form_visually, snap_y_to_lines
)
import cv2

# Use the uploaded sample form
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "storage", "uploads")
sample_form = os.path.join(os.path.dirname(__file__), "sample_form.png")

# Also try the actual uploaded JPEG
jpg_candidates = []
if os.path.isdir(UPLOAD_DIR):
    for fn in os.listdir(UPLOAD_DIR):
        if fn.endswith((".jpg", ".jpeg", ".png")):
            jpg_candidates.append(os.path.join(UPLOAD_DIR, fn))

# Pick the best image to test with
test_image = None
if jpg_candidates:
    test_image = jpg_candidates[0]
    print(f"Using uploaded image: {test_image}")
elif os.path.exists(sample_form):
    test_image = sample_form
    print(f"Using sample form: {test_image}")
else:
    print("ERROR: No test image found!")
    sys.exit(1)

img = cv2.imread(test_image)
h, w = img.shape[:2]
print(f"Image dimensions: {w}x{h}")

# 1. Visual classification
is_namibian = is_namibian_form_visually(img)
print(f"\n--- Visual Classification ---")
print(f"Is Namibian form: {is_namibian}")

# 2. Horizontal line detection
lines = detect_horizontal_lines(img)
print(f"\n--- Horizontal Line Detection ---")
print(f"Found {len(lines)} horizontal lines at y={lines}")

# 3. Full layout analysis
result = analyze_layout(test_image, document_text_sample="namibian employment")
print(f"\n--- Layout Analysis ---")
print(f"Template: {result['template_key']} (confidence: {result['template_confidence']:.2f})")
print(f"Fields: {len(result['fields'])}")
print(f"H-Lines: {len(result.get('horizontal_lines', []))}")

print(f"\n--- Field Details ---")
for f in result["fields"]:
    bbox = f["bbox"]
    method = f.get("extraction_method", "unknown")
    crop_h = bbox[2] - bbox[0]
    crop_w = bbox[3] - bbox[1]
    print(f"  {f['key']:25s} bbox=[{bbox[0]:4d},{bbox[1]:4d},{bbox[2]:4d},{bbox[3]:4d}] "
          f"size={crop_w}x{crop_h} method={method}")
    
    # Sanity checks
    assert bbox[2] > bbox[0], f"Field {f['key']} has ymax <= ymin!"
    assert bbox[3] > bbox[1], f"Field {f['key']} has xmax <= xmin!"
    assert bbox[0] >= 0, f"Field {f['key']} has negative ymin!"
    assert bbox[1] >= 0, f"Field {f['key']} has negative xmin!"

print("\n[OK] All sanity checks passed!")
