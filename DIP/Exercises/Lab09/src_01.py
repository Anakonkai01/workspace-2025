import cv2
import numpy as np

def draw_student_id(image, student_id):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, student_id, (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return image

student_id = "523H0164"

img = cv2.imread('sudoku_original.png', 0)
if img is None:
    print("Error loading sudoku_original.png")
else:
    img_sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    img_sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    
    abs_grad_x = cv2.convertScaleAbs(img_sobel_x)
    abs_grad_y = cv2.convertScaleAbs(img_sobel_y)
    img_sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    img_sobel_color = cv2.cvtColor(img_sobel, cv2.COLOR_GRAY2BGR)
    draw_student_id(img_sobel_color, student_id)
    cv2.imwrite('image_01_01.png', img_sobel_color)

    img_canny = cv2.Canny(img, 50, 150)
    img_canny_color = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)
    draw_student_id(img_canny_color, student_id)
    cv2.imwrite('image_01_02.png', img_canny_color)

    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    img_hough = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
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
            cv2.line(img_hough, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    draw_student_id(img_hough, student_id)
    cv2.imwrite('image_01_03.png', img_hough)

    lines_p = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    img_hough_p = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if lines_p is not None:
        for line in lines_p:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_hough_p, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
    draw_student_id(img_hough_p, student_id)
    cv2.imwrite('image_01_04.png', img_hough_p)

img_circles = cv2.imread('hough_circles_demo_01.png', 0)
if img_circles is None:
    print("Error loading hough_circles_demo_01.png")
else:
    img_circles_color = cv2.cvtColor(img_circles, cv2.COLOR_GRAY2BGR)
    img_blur = cv2.medianBlur(img_circles, 5)
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=55, minRadius=0, maxRadius=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(img_circles_color, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(img_circles_color, (i[0], i[1]), 2, (0, 0, 255), 3)

    draw_student_id(img_circles_color, student_id)
    cv2.imwrite('image_01_05.png', img_circles_color)