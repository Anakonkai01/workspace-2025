import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
from collections import defaultdict



class DetectionFilter:
    """This class is in charge of detect traffic sign"""
    def __init__(self):
        self.blur_ksize
    
    def preprocess_frame(self, frame, median_blur_ksize, clahe_clip_limit, clahe_tile_grid_size):
        """Preprocess frame with blur, median blur, CLAHE, saturation boost on hsv"""

        # apply blur (using median blur to reduce pepper and salt, also keep the edge of the shape)
        frame_processing = cv2.medianBlur(frame, median_blur_ksize)
        
        # Apply clahe on v channel of hsv color
        frame_processing = cv2.cvtColor(frame_processing, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(frame_processing)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
        v_clahe = clahe.apply(v)

        # Boost saturation to enhance color separation
        s = s.astype(np.float32) * self.saturation_boost_factor
        s = np.clip(s, 0, 255).astype(np.uint8)

        # Merge 3 channels and convert back to bgr
        hsv_blur_clahe = cv2.merge([h, s, v_clahe])
        frame_processing = cv2.cvtColor(hsv_blur_clahe, cv2.COLOR_HSV2BGR)
        
        return frame_processing
        
    def color_segmentation(self, frame):
        """Using hsv color to segment with boumding range"""
       
        # convert to hsv 
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
        
        # create range for blue
        lower_blue = np.array([self.blue_lower_h, self.blue_lower_s, self.blue_lower_v]) 
        upper_blue = np.array([self.blue_upper_h, self.blue_upper_s, self.blue_lower_v])

        # create range for red
        lower_red = np.array([self.red_lower_h, self.red_lower_s, self.red_lower_v]) 
        upper_red = np.array([self.red_upper_h, self.red_upper_s, self.red_lower_v])

        # create range for yellow
        lower_yellow = np.array([self.yellow_lower_h, self.yellow_lower_s, self.yellow_lower_v]) 
        upper_yellow = np.array([self.yellow_upper_h, self.yellow_upper_s, self.yellow_lower_v])

        # create mask for each color 
        mask_blue = cv2.inRange()


    def morphology(self, mask, k_size, iter_opening, iter_close):
        """Apply morphological operations"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(k_size, k_size))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=kernel, iterations=iter_opening)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel=kernel, iterations=iter_close)
        
        return mask_clean

    
    def is_bbox_in_roi(self,bbox_coors, roi_params, overlap_threshold=0.5):
        """check if the bbox is in ROI"""
        x, y, w, h = bbox_coors
        roi_x_start, roi_y_start, roi_x_end, roi_y_end = roi_params
        
        x_max = x + w
        y_max = y + h
        
        inter_x1 = max(x, roi_x_start)
        inter_y1 = max(y, roi_y_start)
        inter_x2 = min(x_max, roi_x_end)
        inter_y2 = min(y_max, roi_y_end)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return False
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        bbox_area = w * h
        
        overlap_ratio = inter_area / bbox_area if bbox_area > 0 else 0
        
        return overlap_ratio >= overlap_threshold

        
    def extract_circle_detections(self, mask, roi_params, color_type, min_area=800, max_area=15000)
        """Extract circular detections"""
        
        
        contours, hierrachy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            # area constraint
            if area < min_area or area > max_area:
                continue
            
            # using convexHull to detect circle
            hull = cv2.convexHull(contour)
            perimeter_hull = cv2.arcLength(hull, True) # true is the arcLength is close
            area_hull = cv2.contourArea(hull)

            # get the coors of the rectangle bounding the object
            x, y, w, h = cv2.boundingRect(hull)

            # roi constraint
            is_in_roi = is_bbox_in_roi((x, y, w, h), roi_params, overlap_threshold=0.5)
            
            