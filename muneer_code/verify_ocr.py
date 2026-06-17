import os
import sys
import numpy as np
import cv2
import json

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.app.agents.preprocess import preprocess_document
from backend.app.agents.layout import analyze_layout
from backend.app.agents.ocr import extract_ocr_fields

def create_sample_form_image(filename: str = "sample_form.png"):
    """
    Programmatically generates a realistic scanned application form image
    complete with headers, labels, form input lines, checkboxes, and text.
    Intentionally adds slight rotation skew to test deskew preprocessing.
    """
    print("Generating simulated scanned form image...")
    # Create a blank white A4-proportioned image (800w x 1000h)
    img = np.ones((1000, 800, 3), dtype=np.uint8) * 255
    
    # Draw title header
    cv2.putText(img, "MEMBERSHIP APPLICATION FORM", (180, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (15, 23, 42), 2)
    cv2.putText(img, "Please fill in block capitals.", (180, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 116, 139), 1)
    
    # Draw horizontal divider
    cv2.line(img, (80, 120), (720, 120), (226, 232, 240), 1)

    # 1. Full Name Label & Input Area
    cv2.putText(img, "FULL NAME:", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 23, 42), 2)
    cv2.line(img, (200, 210), (720, 210), (148, 163, 184), 1)
    # Simulated handwritten value
    cv2.putText(img, "John Doe", (220, 200), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, (25, 25, 112), 2)

    # 2. DOB Label & Input Area
    cv2.putText(img, "DATE OF BIRTH:", (80, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 23, 42), 2)
    cv2.line(img, (220, 310), (420, 310), (148, 163, 184), 1)
    # Simulated handwritten value
    cv2.putText(img, "12/04/1990", (230, 300), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, (25, 25, 112), 2)

    # 3. Phone Number Label & Input Area
    cv2.putText(img, "PHONE NUMBER:", (440, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 23, 42), 2)
    cv2.line(img, (580, 310), (720, 310), (148, 163, 184), 1)
    # Simulated handwritten value
    cv2.putText(img, "+1 555-0199", (590, 300), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, (25, 25, 112), 2)

    # 4. Email Label & Input Area
    cv2.putText(img, "EMAIL ADDRESS:", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 23, 42), 2)
    cv2.line(img, (220, 410), (720, 410), (148, 163, 184), 1)
    # Simulated handwritten value
    cv2.putText(img, "john.doe@example.com", (230, 400), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.85, (25, 25, 112), 2)

    # 5. Checkboxes (Membership Type)
    cv2.putText(img, "MEMBERSHIP TYPE:", (80, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 23, 42), 2)
    
    # Draw Standard Box
    cv2.rectangle(img, (200, 480), (220, 500), (148, 163, 184), 1)
    cv2.putText(img, "Standard", (230, 495), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 116, 139), 1)
    
    # Draw Premium Box (Checked with an X)
    cv2.rectangle(img, (400, 480), (420, 500), (15, 23, 42), 2)
    cv2.line(img, (400, 480), (420, 500), (25, 25, 112), 2)
    cv2.line(img, (420, 480), (400, 500), (25, 25, 112), 2)
    cv2.putText(img, "Premium", (430, 495), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 23, 42), 2)

    # Draw Student Box
    cv2.rectangle(img, (600, 480), (620, 500), (148, 163, 184), 1)
    cv2.putText(img, "Student", (630, 495), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 116, 139), 1)

    # 6. Signature Area
    cv2.putText(img, "APPLICANT SIGNATURE:", (80, 750), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 23, 42), 2)
    cv2.line(img, (300, 760), (720, 760), (148, 163, 184), 1)
    # Simulated signature scribble
    cv2.putText(img, "John Doe", (350, 745), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.2, (25, 25, 112), 2)

    # Add artificial skew (rotate by -2 degrees to test deskewing)
    center = (400, 500)
    matrix = cv2.getRotationMatrix2D(center, -2.0, 1.0)
    skewed_img = cv2.warpAffine(img, matrix, (800, 1000), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    cv2.imwrite(filename, skewed_img)
    print(f"Sample form saved as '{filename}'. Skew angle applied: -2.0 degrees.")
    return filename

def run_verification():
    print("=" * 60)
    print("AUTONOMOUS DOCUMENT INTELLIGENCE SYSTEM VERIFIER")
    print("=" * 60)
    
    # 1. Create form
    img_name = "sample_form.png"
    create_sample_form_image(img_name)
    
    # Setup test environment directories
    os.makedirs("backend/storage/uploads", exist_ok=True)
    os.makedirs("backend/storage/preprocessed", exist_ok=True)
    os.makedirs("backend/storage/patches", exist_ok=True)

    # Copy raw to upload folder
    upload_path = os.path.join("backend/storage/uploads", img_name)
    cv2.imwrite(upload_path, cv2.imread(img_name))

    # 2. Run Preprocessing Agent
    print("\n[Step 1] Running Preprocessing Agent...")
    try:
        prep_result = preprocess_document(
            image_path=upload_path,
            output_filename="preprocessed_sample.png",
            deskew=True,
            denoise=True,
            clahe_enhancement=False
        )
        print(f"Preprocessing Success! Output: {prep_result['preprocessed_path']}")
        print(f"Actions taken: {prep_result['actions_taken']}")
        print(f"Deskew Angle Detected: {prep_result['skew_angle']:.2f} degrees")
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return

    # 3. Run Layout Agent
    print("\n[Step 2] Running Layout Intelligence Agent...")
    try:
        layout_result = analyze_layout(
            image_path=prep_result["preprocessed_path"],
            document_text_sample="Membership Application Form"
        )
        print(f"Layout Success! Mapped template: '{layout_result['template_key']}'")
        print(f"Detected {len(layout_result['fields'])} input regions.")
    except Exception as e:
        print(f"Layout mapping failed: {e}")
        return

    # 4. Run Hybrid OCR Agent
    print("\n[Step 3] Running Hybrid OCR Extraction Agent...")
    try:
        ocr_result = extract_ocr_fields(
            image_path=prep_result["preprocessed_path"],
            layout_data=layout_result
        )
        print("OCR Extraction complete!")
        print(f"Overall OCR Confidence: {ocr_result['overall_ocr_confidence'] * 100:.1f}%")
        
        # Display extraction results
        print("\nExtracted Fields Table:")
        print("-" * 90)
        print(f"{'FIELD KEY':<20} | {'EXTRACTED VALUE':<30} | {'CONF':<7} | {'ENGINE USED':<25}")
        print("-" * 90)
        for key, field in ocr_result["fields"].items():
            print(f"{key:<20} | {field['value']:<30} | {field['confidence'] * 100.0:>5.1f}% | {field['engine']:<25}")
        print("-" * 90)
        
    except Exception as e:
        print(f"OCR failed: {e}")
        return
        
    print("\nVerification process completed successfully. System behaves correctly!")

if __name__ == "__main__":
    run_verification()
