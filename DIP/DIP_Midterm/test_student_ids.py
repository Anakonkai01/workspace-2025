"""
Quick test to verify draw_student_ids function works
"""
import cv2
import numpy as np

# Test constant
STUDENT_IDS = "523H0164_523H0177_523H0145"

def draw_student_ids(frame, student_ids=None):
    """Draw student IDs on the top-left corner of the frame"""
    # Use global STUDENT_IDS if not provided
    if student_ids is None:
        student_ids = STUDENT_IDS
    
    # Position in top-left corner
    x, y = 10, 30
    
    # Text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    text_color = (255, 255, 255)  # White
    bg_color = (0, 0, 0)  # Black background
    padding = 5
    
    # Get text size for background rectangle
    (text_w, text_h), baseline = cv2.getTextSize(student_ids, font, font_scale, font_thickness)
    
    # Draw background rectangle
    cv2.rectangle(
        frame,
        (x - padding, y - text_h - padding),
        (x + text_w + padding, y + baseline + padding),
        bg_color,
        -1  # Filled rectangle
    )
    
    # Draw student IDs text
    cv2.putText(
        frame,
        student_ids,
        (x, y),
        font,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA
    )
    
    return frame

# Create a test frame
test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
test_frame[:] = (100, 100, 100)  # Gray background

# Draw student IDs
test_frame = draw_student_ids(test_frame)

# Save test image
cv2.imwrite("test_student_ids.png", test_frame)

print("✓ Test image created: test_student_ids.png")
print("✓ Student IDs should be visible in the top-left corner")
print(f"✓ Text: {STUDENT_IDS}")
