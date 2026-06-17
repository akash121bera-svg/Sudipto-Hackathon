"""
Precise bbox calibration for the 768x889 Namibian Employment Application form.

Measured pixel coordinates from the overlay image (field_overlay_diag.png).
Format: [ymin, xmin, ymax, xmax] in absolute pixels.

Image size: 768w x 889h

Findings from visual inspection:
- position_applied:    y=200-248, x=10-368   (value-only zone, under label row)
- office_ministry:     y=200-248, x=370-760  (right side of same row)
- duty_station:        y=252-296, x=10-368   (under "3. Duty Station" label)
- assume_duty:         y=226-296, x=370-760  (right side, 4 & 5 row)
- full_name_surname:   y=316-356, x=180-540  (1a. Surname row)
- full_name:           y=390-430, x=180-555  (2. First names row — actual value)
- namibian_id:         y=432-468, x=10-270   (3. Namibian Identity Number row)
- date_of_birth:       y=432-468, x=380-555  (4. Date of birth row)
- citizenship:         y=490-525, x=10-270   (5. Passport/6. Citizenship row — citizenship only)
- postal_address:      y=530-590, x=10-270   (7. Postal Address rows)
- residential_address: y=530-590, x=380-555  (8. Residential Address rows)
- phone_number:        y=595-630, x=10-200   (10. Contact details, Home No.)
- email_address:       y=610-648, x=210-555  (10. Email row)

To compute normalized [ymin, xmin, ymax, xmax] (scale to 1000):
  norm_y = pixel_y / 889 * 1000
  norm_x = pixel_x / 768 * 1000

Printing normalized values for layout.py template:
"""

W, H = 768, 889

fields_px = {
    "position_applied":    (200, 10,  248, 368),
    "office_ministry":     (200, 370, 248, 760),
    "duty_station":        (252, 10,  296, 368),
    "assume_duty":         (226, 370, 296, 760),
    "full_name":           (390, 180, 430, 555),
    "namibian_id":         (432, 10,  470, 270),
    "date_of_birth":       (432, 380, 470, 555),
    "citizenship":         (490, 10,  525, 270),
    "postal_address":      (530, 10,  592, 270),
    "residential_address": (530, 380, 592, 555),
    "phone_number":        (596, 10,  630, 200),
    "email_address":       (610, 10,  648, 555),
}

print("Normalized bboxes for layout.py (scale: 1000):")
print("# [ymin, xmin, ymax, xmax]")
for key, (ymin, xmin, ymax, xmax) in fields_px.items():
    n_ymin = round(ymin / H * 1000)
    n_xmin = round(xmin / W * 1000)
    n_ymax = round(ymax / H * 1000)
    n_xmax = round(xmax / W * 1000)
    print(f'  {{\"key\": \"{key}\", \"bbox\": [{n_ymin}, {n_xmin}, {n_ymax}, {n_xmax}], "label": ...}},')

print()
print("Absolute px sizes:")
for key, (ymin, xmin, ymax, xmax) in fields_px.items():
    print(f"  {key:25s}: {xmax-xmin}w x {ymax-ymin}h")
