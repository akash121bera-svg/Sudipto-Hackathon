"""
End-to-end OCR test: layout analysis -> field cropping -> OCR extraction.
Tests the full integrated pipeline on the uploaded Namibian form.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["ENABLE_MOCK_FALLBACK"] = "false"

from backend.app.agents.layout import analyze_layout
from backend.app.agents.ocr import extract_ocr_fields

# Find the uploaded image
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "storage", "uploads")
test_image = None
for fn in os.listdir(UPLOAD_DIR):
    if fn.endswith((".jpg", ".jpeg", ".png")):
        test_image = os.path.join(UPLOAD_DIR, fn)
        break

if not test_image:
    print("No test image found!")
    sys.exit(1)

print(f"Image: {test_image}")
print("="*80)

# Step 1: Layout analysis
print("\n[1/2] Running layout analysis...")
layout_result = analyze_layout(test_image, document_text_sample="namibian employment")
print(f"Template: {layout_result['template_key']}")
print(f"Fields: {len(layout_result['fields'])}")
print(f"H-Lines: {len(layout_result.get('horizontal_lines', []))}")

# Step 2: OCR extraction  
print("\n[2/2] Running OCR extraction (this may take 30-60 seconds)...")
ocr_result = extract_ocr_fields(test_image, layout_result)

print("\n" + "="*80)
print("EXTRACTION RESULTS")
print("="*80)

for key, field in ocr_result["fields"].items():
    conf_bar = "#" * int(field["confidence"] * 20)
    print(f"\n  {key:25s}: {field['value']!r}")
    print(f"  {'':25s}  conf={field['confidence']:.3f} [{conf_bar}]")
    print(f"  {'':25s}  engine={field['engine']}  method={field.get('extraction_method', 'n/a')}")

print(f"\n{'='*80}")
print(f"Overall OCR Confidence: {ocr_result['overall_ocr_confidence']:.3f}")
print(f"{'='*80}")
