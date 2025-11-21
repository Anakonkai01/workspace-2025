import cv2
import numpy as np

def draw_student_id(image, student_id):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, student_id, (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return image

student_id = "523H0164"

img = cv2.imread('sudoku_original.png') 
if img is None:
    print("Error loading sudoku_original.png")
else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    draw_student_id(img, student_id)
    cv2.imwrite('image_02_01.png', img)