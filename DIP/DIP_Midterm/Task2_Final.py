import cv2
import numpy as np
import matplotlib.pyplot as plt

#======================================HELPER FUNCTION CLASS======================================
class Utitiles:
    def __init__(self):
        pass
    
    def filter_contours_by_coords(self, contours, x_range, y_range, w_range=None, h_range=None):
        '''Function for filtering the contours by coordinates''' 
        filter_contours = []
        
        # Iterative over contours to check contour satisfy coordinate range
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if x >= x_range[0] and x + w <= x_range[1] and y >= y_range[0] and y + h <= y_range[1]:
                if w_range is not None:
                    if not (w >= w_range[0] and w <= w_range[1]):
                        continue
                if h_range is not None:
                    if not (h >= h_range[0] and h <= h_range[1]):
                        continue
                filter_contours.append(contour)
        return filter_contours

    def find_and_draw_contours(self, image, min_area=200, max_area=2000, contours_to_draw=None):
        '''Function to find contours of image and draw contours in that image''' 
        if contours_to_draw is None:
            # Find contours if not provided
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours = contours_to_draw
            
        # Convert to BGR to draw colored bounding boxes
        image_with_boxes = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if contours_to_draw is None:
            # Filter contours by area if they were found in this function call
            contours_to_process = [c for c in contours if cv2.contourArea(c) > min_area and cv2.contourArea(c) < max_area]
        else:
            # Use provided contours
            contours_to_process = contours_to_draw

        for contour in contours_to_process:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image_with_boxes, (x, y), (x + w , y + h), (0, 255, 0), 2)

        return contours, image_with_boxes
 
    def showimage(self, image, window_name):
        # Function to display the image
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#======================================MAIN LOGIC======================================
if __name__ == "__main__":
    utils = Utitiles()

    # Thresholding values
    THRESHOLD_NOISE_PART = 30
    THRESHOLD_PART_1 = 160
    THRESHOLD_PART_2 = 100
    THRESHOLD_PART_3 = 170

    # Noise Part width and height bounding boxes conditions for filtering small noise
    NOISE_MIN_WH = 20 
    NOISE_MAX_WH = 100 

    # Reading image & prepocessing 
    digits_img = cv2.imread("input.png")
    if digits_img is None:
        print("Error: Can't not find the the input image")
        exit()
        
    gray_digits_img = cv2.cvtColor(digits_img, cv2.COLOR_BGR2GRAY)


    '''I. Noise Part Handling And Find Contours'''
    # Small point has same color with background to separate digits 8 and 9 in the gray input image 
    gray_digits_img[530: 540, 330: 340] = 255

    # Noise region coordinates
    Y_START_NOISE = 265
    Y_END_NOISE = gray_digits_img.shape[0] # Max image height
    X_START_NOISE = 250
    X_END_NOISE = 500
    noise_part = gray_digits_img[Y_START_NOISE: Y_END_NOISE, X_START_NOISE: X_END_NOISE].copy()

    # Binarization 
    condition = (noise_part > THRESHOLD_NOISE_PART) & (noise_part != 0)
    noise_part[condition] = 0
    noise_part[~condition] = 255  

    # Morphological Operations (Opening and Dilate)
    opening_local_region = cv2.morphologyEx(noise_part, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    noise_part_processed = cv2.dilate(opening_local_region, np.ones((3, 3), np.uint8))

    # Find and filter contours
    contours_noise_part_all, _ = utils.find_and_draw_contours(noise_part_processed) 

    # Filter contours by width and height to remove small noise
    contours_noise_part = utils.filter_contours_by_coords(
        contours_noise_part_all, 
        (0, gray_digits_img.shape[1]),
        (0, gray_digits_img.shape[0]), 
        (NOISE_MIN_WH, NOISE_MAX_WH), #For width range
        (NOISE_MIN_WH, NOISE_MAX_WH) # For height range
    )  
    color_noise_part = cv2.cvtColor(noise_part, cv2.COLOR_GRAY2BGR)
    for contour in contours_noise_part:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(color_noise_part, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    '''II. Other Part Handling And Find Contours'''
    gray_digits_img_2 = gray_digits_img.copy()
    
    #Coordiante for ROI other parts
    COORD_1 = 265 
    COORD_2 = 250 
    COORD_3 = 500 
    # Define Regions of Interest (ROI)
    part_1 = gray_digits_img_2[:COORD_1, :]       # Above the noise region
    part_2 = gray_digits_img_2[COORD_2:, :COORD_1] # Left of the noise region
    part_3 = gray_digits_img_2[:, COORD_3:]       # Right of the noise region

    #Connecting the half-cut digits on the right side of the image (For part_3 handling)
    part_3[435:455, 78: 83] = 0

    # --- LOGIC FOR PART 1 ---
    # Binarization
    _, bin_part_1 = cv2.threshold(part_1, THRESHOLD_PART_1, 255, cv2.THRESH_BINARY)
    inv_bin_part_1 = cv2.bitwise_not(bin_part_1)

    # Find Contours
    contours_part_1, _ = cv2.findContours(inv_bin_part_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color_part_1 = cv2.cvtColor(inv_bin_part_1, cv2.COLOR_GRAY2BGR)
    for contour in contours_part_1:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(color_part_1, (x, y), (x + w, y + h), (0, 255, 0), 2)
    

    # --- LOGIC FOR PART 2 ---
    # Binarization
    _, bin_part_2 = cv2.threshold(part_2, THRESHOLD_PART_2, 255, cv2.THRESH_BINARY)
    inv_bin_part_2 = cv2.bitwise_not(bin_part_2)

    color_part_2 = cv2.cvtColor(inv_bin_part_2, cv2.COLOR_GRAY2BGR)
    
    # Find initial contours and remove straight lines
    contours_part_2, _ = cv2.findContours(inv_bin_part_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours_part_2:
        (x, y, w, h) = cv2.boundingRect(contour)
        # Check if contour spans the entire width (likely a horizontal line)
        if w == inv_bin_part_2.shape[1]:
            # Set the area corresponding to the line to black (0)
            color_part_2[y:y+h, x:x+w] = 0

    # Apply Dilate and find new contours
    part_2_gray = cv2.cvtColor(color_part_2, cv2.COLOR_BGR2GRAY)
    inv_bin_part_with_dialte = cv2.dilate(part_2_gray, np.ones((5, 5), np.uint8))
    contours_after_dialte, _ = cv2.findContours(inv_bin_part_with_dialte, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color = cv2.cvtColor(inv_bin_part_with_dialte, cv2.COLOR_GRAY2BGR) 
    for contour in contours_after_dialte:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(color, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Search for special bounding boxes (likely digit "4")
    bbox_for_part_2 = None
    for contour in contours_after_dialte:
        (x, y, w, h) = cv2.boundingRect(contour)
        # Look for digit "4"
        if x == 0 and (y > 110 and y < 220):
            bbox_for_part_2 = (x, y, w, h)

    # --- LOGIC FOR PART 3 ---
    # Binarization
    _, bin_part_3 = cv2.threshold(part_3, THRESHOLD_PART_3, 255, cv2.THRESH_BINARY)
    inv_bin_part_3 = cv2.bitwise_not(bin_part_3)

    # Morphological Operations (Opening and Erode)
    inv_bin_part_3 = cv2.morphologyEx(inv_bin_part_3, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    inv_bin_part_3 = cv2.erode(inv_bin_part_3, np.ones((3, 3), np.uint8))

    # Find Contours
    contours_part_3, _ = cv2.findContours(inv_bin_part_3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_color_part_3 = cv2.cvtColor(inv_bin_part_3, cv2.COLOR_GRAY2BGR)
 
    '''IV. Apply Contours to Original Image'''
    
    # Offset values to map ROI coordinates back to the original image
    OFFSET_X_NOISE_PART = X_START_NOISE # 250
    OFFSET_Y_NOISE_PART = Y_START_NOISE # 265
    OFFSET_X_PART2 = 0
    OFFSET_Y_PART2 = COORD_2 # 250
    OFFSET_X_PART3 = COORD_3 # 500
    OFFSET_Y_PART3 = 0

    # Area conditions
    AREA_MIN_PART_1 = 200
    AREA_MAX_PART_1 = 1500
    
    AREA_MIN_PART_2 = 300
    AREA_MAX_PART_2 = 1500

    AREA_MIN_PART_3 = 200
    AREA_MAX_PART_3 = 1500

    # Part 1: Draw bounding boxes on the original image
    for contour in contours_part_1:
        if cv2.contourArea(contour) > AREA_MIN_PART_1 and cv2.contourArea(contour) < AREA_MAX_PART_1:
            x, y, w, h = cv2.boundingRect(contour)
            # No offset needed for Part 1 (starts at 0,0)
            cv2.rectangle(digits_img, (x, y), (x + w , y + h), (0, 255, 0), 2)
    # Noise Part: Draw bounding boxes with offset
    for contour in contours_noise_part:
        x, y, w, h = cv2.boundingRect(contour)
        x = x + OFFSET_X_NOISE_PART
        y = y + OFFSET_Y_NOISE_PART
        cv2.rectangle(digits_img, (x, y), (x + w , y + h), (0, 255, 0), 2)
    # Part 2: Draw bounding boxes with offset
    for contour in contours_part_2:
        area = cv2.contourArea(contour)
        if area > AREA_MIN_PART_2 and area < AREA_MAX_PART_2:
            x, y, w, h = cv2.boundingRect(contour)
            # Remove bounding boxes of horizonal lines (already filtered in logic, but safe to check again)
            if w == part_2.shape[1]:
                continue
                
            x = x + OFFSET_X_PART2
            y = y + OFFSET_Y_PART2
            cv2.rectangle(digits_img, (x, y),(x+w, y+h), (0, 255, 0), 2)
    # Applying for digit "4" in part 2 (bbox_for_part_2)
        (x, y, w, h) = bbox_for_part_2
        x = x + OFFSET_X_PART2
        y = y + OFFSET_Y_PART2
        cv2.rectangle(digits_img, (x, y),(x+w, y+h), (0, 255, 0), 2)
    # Part 3: Draw bounding boxes with offset
    for contour in contours_part_3:
        if cv2.contourArea(contour) > AREA_MIN_PART_3 and cv2.contourArea(contour) < AREA_MAX_PART_3:
            x, y, w, h = cv2.boundingRect(contour)
            if w == part_3.shape[1]:
                continue
            x = x + OFFSET_X_PART3
            y = y + OFFSET_Y_PART3
            cv2.rectangle(digits_img, (x, y), (x + w , y + h), (0, 255, 0), 2)


    utils.showimage(digits_img, "Final Result")
    cv2.imwrite("Task2_output.png", digits_img)