import cv2
import numpy as np

def draw_student_id(image, student_id):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, student_id, (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return image

student_id = "523H0164" 

img = cv2.imread('barcode.jpg')
if img is None:
    print("Error loading barcode.jpg")
else:
    bardet = cv2.barcode_BarcodeDetector()

    ok, decoded_info, decoded_type, corners = bardet.detectAndDecodeWithType(img)
    
    if ok:
        for i, code in enumerate(decoded_info):
            text = "{} ({})".format(code, decoded_type[i])
            points = corners[i].astype(int)
            
            cv2.polylines(img, [points], True, (0, 255, 0), 2)
            
            cv2.putText(img, text, (points[0][0], points[0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    draw_student_id(img, student_id)
    cv2.imwrite('image_03_01.png', img)
    print("Done! Output saved to image_03_01.png")