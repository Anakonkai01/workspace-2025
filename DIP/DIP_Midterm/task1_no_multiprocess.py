import cv2 
import numpy as np 
import time
from collections import defaultdict

# =============================================================================
# GLOBAL VARIABLES (Parameters)
# =============================================================================

# --- Blue Color Parameters ---
blue_lower_h = 102
blue_upper_h = 144
blue_lower_s = 216 
blue_upper_s = 255
blue_lower_v = 81
blue_upper_v = 227
blue_ksize = 7
blue_open_iter = 1
blue_close_iter = 5
blue_clahe_clip_limit = 30 
blue_blur_ksize = 7

# --- Red Color Parameters ---
red_lower_h = 117
red_upper_h = 179
red_lower_s = 40
red_upper_s = 255
red_lower_v = 0
red_upper_v = 255
red_ksize = 2
red_open_iter = 2
red_close_iter = 5
red_clahe_clip_limit = 30
red_blur_ksize = 5

# --- Yellow Color Parameters ---
yellow_lower_h = 8
yellow_upper_h = 18
yellow_lower_s = 111 
yellow_upper_s = 255
yellow_lower_v = 100
yellow_upper_v = 255
yellow_ksize = 3
yellow_open_iter = 1
yellow_close_iter = 5
yellow_clahe_clip_limit = 30
yellow_blur_ksize = 7

# --- Area Parameters ---
CIRCLE_MIN_AREA = 300
CIRCLE_MAX_AREA = 15000
CIRCLE_TRUST_THRESHOLD = 725  # Circles >= this area are trusted (bypass ROI check)
TRIANGLE_MIN_AREA = 400
TRIANGLE_MAX_AREA = 50000
TRIANGLE_TRUST_THRESHOLD = 1500  # Triangles >= this area are trusted (bypass ROI check)

# --- Shape Quality Parameters ---
TRIANGLE_MIN_SOLIDITY = 0.75  # Minimum solidity (area/hull_area) for triangle detection
TRIANGLE_EPSILON_FACTOR = 0.03  # Contour approximation accuracy (% of perimeter)
TRIANGLE_MAX_VERTICES = 7  # Maximum vertices for triangle approximation
CIRCLE_SMALL_CIRCULARITY = 0.87  # Circularity threshold for small circles
CIRCLE_LARGE_CIRCULARITY = 0.93  # Circularity threshold for large circles

# --- Image Processing Parameters ---
CLAHE_CLIP_DIVISOR = 10.0  # Divisor for CLAHE clip limit normalization
CLAHE_TILE_GRID_SIZE = (1, 1)  # Tile grid size for CLAHE (adaptive histogram equalization)
SATURATION_BOOST_FACTOR = 1.5  # Multiplication factor for saturation enhancement

# --- ROI Parameters ---
#  (x_start, y_start, x_end, y_end) in percentage of full frame
BLUE_ROI = (0.4, 0.0, 0.7, 0.475)      
RED_ROI = (0.45, 0.2, 1.0, 0.445)      
YELLOW_ROI = (0.45, 0.2375, 0.8, 0.5) 

# --- Temporal Filtering Parameters  ---
# Blue signs temporal filtering
BLUE_MIN_DURATION_SEC = 2.0
BLUE_MAX_GAP_SEC = 0.5
BLUE_IOU_THRESHOLD = 0.3

# Red signs temporal filtering
RED_MIN_DURATION_SEC = 2.0
RED_MAX_GAP_SEC = 0.5
RED_IOU_THRESHOLD = 0.3

# Yellow signs temporal filtering
YELLOW_MIN_DURATION_SEC = 3.0
YELLOW_MAX_GAP_SEC = 0.5
YELLOW_IOU_THRESHOLD = 0.3

# --- Processing Limit ---
MAX_FRAME_ID = 2800  # Stop detection at this frame

# --- Debug Mode Toggle ---
DEBUG_MODE = False  # Set to False to disable ROI boxes and metrics overlay

# --- File Paths ---
INPUT_VIDEO_PATH = 'task1.mp4'
OUTPUT_VIDEO_PATH = 'task1_output.mp4'

print(f"Debug mode: {'ENABLED' if DEBUG_MODE else 'DISABLED'}")


# =============================================================================
# TEMPORAL FILTER CLASS 
# =============================================================================

class TemporalSignFilter:
    """Filter traffic signs based on temporal consistency with color-specific parameters"""
    
    def __init__(self, fps, color_params=None):
        """
        Args:
            fps: Frames per second of the video
            color_params: Dict mapping color -> (min_duration_sec, max_gap_sec, iou_threshold)
                         Example: {'blue': (2.0, 0.5, 0.3), 'red': (1.5, 0.7, 0.25)}
        """
        self.fps = fps
        
        # Store color-specific parameters
        self.min_frames = {}
        self.max_gap_frames = {}
        self.iou_thresholds = {}
        
        if color_params:
            for color, (min_dur, max_gap, iou_thresh) in color_params.items():
                self.min_frames[color] = int(min_dur * fps)
                self.max_gap_frames[color] = int(max_gap * fps)
                self.iou_thresholds[color] = iou_thresh
        
        self.tracks = defaultdict(list)
        self.next_track_id = 0
        
        # Cache for validated detections 
        self._validated_cache = {}
        self._cache_built = False
        
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        x1_max, y1_max = x1 + w1, y1 + h1
        x2_max, y2_max = x2 + w2, y2 + h2
        
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)
        
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def add_detections(self, frame_num, detections):
        """Add detections from current frame with color-specific IoU threshold"""
        for detection_data in detections:
            bbox, color = detection_data[0], detection_data[1]
            # Extract metrics if available (for debugging)
            metrics = detection_data[2] if len(detection_data) > 2 else {}
            
            # Use color-specific IoU threshold
            iou_threshold = self.iou_thresholds.get(color, 0.3)
            max_gap = self.max_gap_frames.get(color, int(0.5 * self.fps))
            
            best_match_id = None
            best_iou = 0
            
            for track_id, track_data in self.tracks.items():
                if len(track_data) == 0:
                    continue
                    
                last_detection = track_data[-1]
                
                if (last_detection['color'] == color and 
                    frame_num - last_detection['frame'] <= max_gap):
                    
                    iou = self.calculate_iou(bbox, last_detection['bbox'])
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_match_id = track_id
            
            if best_match_id is not None:
                self.tracks[best_match_id].append({
                    'frame': frame_num,
                    'bbox': bbox,
                    'color': color,
                    'metrics': metrics
                })
            else:
                self.tracks[self.next_track_id] = [{
                    'frame': frame_num,
                    'bbox': bbox,
                    'color': color,
                    'metrics': metrics
                }]
                self.next_track_id += 1
    
    def interpolate_missing_frames(self, track_data, color):
        """Fill missing frames with linear interpolation using color-specific max gap"""
        if len(track_data) < 2:
            return track_data
        
        # Get color-specific max gap
        max_gap = self.max_gap_frames.get(color, int(0.5 * self.fps))
        
        interpolated = []
        
        for i in range(len(track_data) - 1):
            current = track_data[i]
            next_det = track_data[i + 1]
            
            interpolated.append(current)
            
            frame_gap = next_det['frame'] - current['frame']
            
            if 1 < frame_gap <= max_gap:
                x1, y1, w1, h1 = current['bbox']
                x2, y2, w2, h2 = next_det['bbox']
                
                for j in range(1, frame_gap):
                    alpha = j / frame_gap
                    
                    x_interp = int(x1 + (x2 - x1) * alpha)
                    y_interp = int(y1 + (y2 - y1) * alpha)
                    w_interp = int(w1 + (w2 - w1) * alpha)
                    h_interp = int(h1 + (h2 - h1) * alpha)
                    
                    interpolated.append({
                        'frame': current['frame'] + j,
                        'bbox': (x_interp, y_interp, w_interp, h_interp),
                        'color': current['color'],
                        'metrics': current.get('metrics', {}),
                        'interpolated': True
                    })
        
        interpolated.append(track_data[-1])
        return interpolated
    
    def smooth_bounding_boxes(self, track_data, window_size=5):
        """Smooth bounding boxes with moving average"""
        if len(track_data) < window_size:
            return track_data
        
        smoothed = []
        
        for i in range(len(track_data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(track_data), i + window_size // 2 + 1)
            
            bboxes = [d['bbox'] for d in track_data[start_idx:end_idx]]
            
            x_avg = int(np.mean([b[0] for b in bboxes]))
            y_avg = int(np.mean([b[1] for b in bboxes]))
            w_avg = int(np.mean([b[2] for b in bboxes]))
            h_avg = int(np.mean([b[3] for b in bboxes]))
            
            smoothed_det = track_data[i].copy()
            smoothed_det['bbox'] = (x_avg, y_avg, w_avg, h_avg)
            smoothed.append(smoothed_det)
        
        return smoothed
    
    def build_detection_cache(self):
        """
        Uses color-specific minimum duration for validation
        """
        print("   Building detection cache...")
        start = time.time()
        
        for track_id, track_data in self.tracks.items():
            if len(track_data) == 0:
                continue
            
            # Get color from first detection
            color = track_data[0]['color']
            
            # Use color-specific minimum frames
            min_frames_for_color = self.min_frames.get(color, int(2.0 * self.fps))
            
            first_frame = track_data[0]['frame']
            last_frame = track_data[-1]['frame']
            duration_frames = last_frame - first_frame + 1
            
            if duration_frames >= min_frames_for_color:
                interpolated = self.interpolate_missing_frames(track_data, color)
                smoothed = self.smooth_bounding_boxes(interpolated, window_size=5)
                
                for detection in smoothed:
                    frame_num = detection['frame']
                    if frame_num not in self._validated_cache:
                        self._validated_cache[frame_num] = []
                    self._validated_cache[frame_num].append(
                        (detection['bbox'], detection['color'], detection.get('metrics', {}))
                    )
        
        self._cache_built = True
        print(f"   Cache built in {time.time() - start:.2f}s ({len(self._validated_cache)} frames)")
    
    def get_validated_detections(self, frame_num):
        """Get validated detections (using cache)"""
        if not self._cache_built:
            self.build_detection_cache()
        
        return self._validated_cache.get(frame_num, [])
    
    def get_statistics(self):
        """Get tracking statistics with color-specific minimum frames"""
        total_tracks = len(self.tracks)
        valid_tracks = 0
        
        for track in self.tracks.values():
            if len(track) > 0:
                # Get color from first detection
                color = track[0]['color']
                # Use color-specific minimum frames
                min_frames_for_color = self.min_frames.get(color, int(2.0 * self.fps))
                
                duration_frames = track[-1]['frame'] - track[0]['frame'] + 1
                if duration_frames >= min_frames_for_color:
                    valid_tracks += 1
        
        return total_tracks, valid_tracks


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def morphology(mask, k_size, iter_opening, iter_close):
    """Apply morphological operations"""
    k_size = max(1, k_size)
    if k_size % 2 == 0:
        k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(k_size, k_size)) 
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iter_opening)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=iter_close)
    return mask_clean

def preprocess_frame(frame, clip_limit, blur_ksize):
    """Preprocess frame with median blur, HSV conversion, CLAHE, and saturation boost"""
    blur_ksize = max(3, blur_ksize)
    if blur_ksize % 2 == 0:
        blur_ksize += 1
        
    frame_processing = cv2.medianBlur(frame, blur_ksize) 
    frame_processing = cv2.cvtColor(frame_processing, cv2.COLOR_BGR2HSV)      
    
    h, s, v = cv2.split(frame_processing)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit / CLAHE_CLIP_DIVISOR, 
                            tileGridSize=CLAHE_TILE_GRID_SIZE)
    v_clahe = clahe.apply(v)
    
    # Boost saturation to enhance color separation
    s = s.astype(np.float32) * SATURATION_BOOST_FACTOR
    s = np.clip(s, 0, 255).astype(np.uint8)
    
    hsv_blur_clahe = cv2.merge([h, s, v_clahe])
    return hsv_blur_clahe

def convert_roi_to_pixels(roi_percent, w_full, h_full):
    """Convert ROI from percentage to pixel coordinates"""
    x_start_pct, y_start_pct, x_end_pct, y_end_pct = roi_percent
    return (
        int(w_full * x_start_pct),
        int(h_full * y_start_pct),
        int(w_full * x_end_pct),
        int(h_full * y_end_pct)
    )

def is_bbox_in_roi(bbox, roi_params, overlap_threshold=0.5):
    """Check if bbox is in ROI"""
    x, y, w, h = bbox
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


# =============================================================================
# DETECTION FUNCTIONS (WITH METRICS)
# =============================================================================

def extract_circle_detections(mask, roi_params, color_type, min_area=300, max_area=15000):
    """Extract circular detections with two-layer ROI logic"""
    detections = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour) 
        
        if area < min_area or area > max_area:
            continue
        
        hull = cv2.convexHull(contour)
        perimeter_hull = cv2.arcLength(hull, True) 
        area_hull = cv2.contourArea(hull)
        x, y, w, h = cv2.boundingRect(hull)
        
        is_in_roi = is_bbox_in_roi((x, y, w, h), roi_params, overlap_threshold=0.5)
        
        # TWO-LAYER ROI LOGIC for circles:
        # Layer 1: Filter small noise outside ROI
        if area < CIRCLE_TRUST_THRESHOLD and not is_in_roi:
            continue  # Reject small circles outside ROI (noise)
        # Layer 2: Trust large circles, bypass ROI check
        elif area >= CIRCLE_TRUST_THRESHOLD:
            pass  # Allow large circles regardless of ROI (better tracking)
        # Layer 1: Allow small circles inside ROI
        elif area < CIRCLE_TRUST_THRESHOLD and is_in_roi:
            pass  # Allow small circles inside ROI
        
        if perimeter_hull > 0:
            circularity = 4 * np.pi * (area_hull / (perimeter_hull * perimeter_hull))
            
            if area < CIRCLE_TRUST_THRESHOLD:
                circularity_param = CIRCLE_SMALL_CIRCULARITY
            else:
                circularity_param = CIRCLE_LARGE_CIRCULARITY

            if circularity > circularity_param:
                # Store bbox, color, and metrics
                metrics = {
                    'area': int(area_hull),
                    'circularity': round(circularity, 3),
                    'shape': 'circle'
                }
                detections.append(((x, y, w, h), color_type, metrics))
                
    return detections

def extract_triangle_detections(mask, roi_params, color_type, min_area=825, max_area=50000):
    """Extract triangular detections with two-layer ROI logic"""
    detections = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < min_area or area > max_area: 
            continue
            
        x, y, w, h = cv2.boundingRect(contour)

        is_in_roi = is_bbox_in_roi((x, y, w, h), roi_params, overlap_threshold=0.5)
        
        # TWO-LAYER ROI LOGIC for triangles:
        # Layer 1: Filter small noise outside ROI
        if area < TRIANGLE_TRUST_THRESHOLD and not is_in_roi:
            continue  # Reject small triangles outside ROI (noise)
        # Layer 2: Trust large triangles, bypass ROI check
        elif area >= TRIANGLE_TRUST_THRESHOLD:
            pass  # Allow large triangles regardless of ROI (better tracking)
        # Layer 1: Allow small triangles inside ROI
        elif area < TRIANGLE_TRUST_THRESHOLD and is_in_roi:
            pass  # Allow small triangles inside ROI
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        if hull_area == 0:
            continue
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        solidity = float(area) / hull_area
        
        if solidity <= TRIANGLE_MIN_SOLIDITY:
            continue

        # Approximate contour to polygon with epsilon proportional to perimeter
        epsilon = TRIANGLE_EPSILON_FACTOR * perimeter 
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)
        
        # Filter for triangular shapes (allow slight over-segmentation)
        if num_vertices <= TRIANGLE_MAX_VERTICES:
            # Store bbox, color, and metrics
            metrics = {
                'area': int(area),
                'solidity': round(solidity, 3),
                'shape': 'triangle'
            }
            detections.append(((x, y, w, h), color_type, metrics))
            
    return detections

def draw_detections_with_metrics(frame, detections, debug_mode=True):
    """Draw bounding boxes with optional area and circularity/solidity metrics"""
    color_map = {
        'blue': (0, 255, 0),
        'red': (0, 255, 0),
        'yellow': (0, 255, 0)
    }
    
    for detection_data in detections:
        bbox = detection_data[0]
        color_type = detection_data[1]
        metrics = detection_data[2] if len(detection_data) > 2 else {}
        
        x, y, w, h = bbox
        color_bgr = color_map.get(color_type, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 3)
        
        # Draw metrics above the bounding box (only if debug mode is enabled)
        if debug_mode and metrics:
            shape = metrics.get('shape', 'unknown')
            area = metrics.get('area', 0)
            
            if shape == 'circle':
                circularity = metrics.get('circularity', 0)
                text = f"A:{area} C:{circularity:.2f}"
            else:  # triangle
                solidity = metrics.get('solidity', 0)
                text = f"A:{area} S:{solidity:.2f}"
            
            # Position text above the box
            text_y = max(y - 10, 20)
            
            # Draw text background
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, text_y - text_h - 5), (x + text_w + 5, text_y + 5), (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(frame, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return frame

def draw_roi_boxes(frame, roi_params_dict):
    """Draw ROI rectangles for each color"""
    roi_colors = {
        'blue': (255, 0, 0),    # Blue
        'red': (0, 0, 255),      # Red
        'yellow': (0, 255, 255)  # Yellow
    }
    
    for color_name, roi_params in roi_params_dict.items():
        x_start, y_start, x_end, y_end = roi_params
        color_bgr = roi_colors.get(color_name, (255, 255, 255))
        
        # Draw ROI rectangle
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color_bgr, 2)
        
        # Draw label
        label = f"{color_name.upper()} ROI"
        cv2.putText(frame, label, (x_start + 10, y_start + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2, cv2.LINE_AA)
    
    return frame

def draw_frame_id(frame, frame_num):
    """Draw frame number on the video"""
    text = f"Frame: {frame_num}"
    
    # Position in top-left corner
    x, y = 10, 30
    
    # Draw text background
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.rectangle(frame, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    
    return frame


# =============================================================================
# DETECTION PROCESSING FUNCTION
# =============================================================================

def process_single_frame(frame_full, frame_num, height_new, width_new, roi_params_dict):
    """
    Process a single frame for detection
    Returns list of detections for this frame
    """
    # Crop frame for processing (top 47.5%)
    frame_to_process = frame_full[0:height_new, 0:width_new]
    
    # Process BLUE
    hsv_blue = preprocess_frame(frame_to_process, blue_clahe_clip_limit, blue_blur_ksize)
    lower_blue = np.array([blue_lower_h, blue_lower_s, blue_lower_v])
    upper_blue = np.array([blue_upper_h, blue_upper_s, blue_upper_v])
    mask_blue = cv2.inRange(hsv_blue, lower_blue, upper_blue)
    mask_blue_clean = morphology(mask_blue, blue_ksize, blue_open_iter, blue_close_iter)
    
    # Process RED
    hsv_red = preprocess_frame(frame_to_process, red_clahe_clip_limit, red_blur_ksize)
    lower_red = np.array([red_lower_h, red_lower_s, red_lower_v])
    upper_red = np.array([red_upper_h, red_upper_s, red_upper_v])
    mask_red = cv2.inRange(hsv_red, lower_red, upper_red)
    mask_red_clean = morphology(mask_red, red_ksize, red_open_iter, red_close_iter)
    
    # Process YELLOW
    hsv_yellow = preprocess_frame(frame_to_process, yellow_clahe_clip_limit, yellow_blur_ksize)
    lower_yellow = np.array([yellow_lower_h, yellow_lower_s, yellow_lower_v])
    upper_yellow = np.array([yellow_upper_h, yellow_upper_s, yellow_upper_v])
    mask_yellow = cv2.inRange(hsv_yellow, lower_yellow, upper_yellow)
    mask_yellow_clean = morphology(mask_yellow, yellow_ksize, yellow_open_iter, yellow_close_iter)
    
    # Extract detections (with metrics)
    all_detections = []
    all_detections.extend(extract_circle_detections(mask_blue_clean, roi_params_dict['blue'], 'blue', 
                                                     CIRCLE_MIN_AREA, CIRCLE_MAX_AREA))
    all_detections.extend(extract_circle_detections(mask_red_clean, roi_params_dict['red'], 'red', 
                                                     CIRCLE_MIN_AREA, CIRCLE_MAX_AREA))
    all_detections.extend(extract_triangle_detections(mask_yellow_clean, roi_params_dict['yellow'], 'yellow', 
                                                       TRIANGLE_MIN_AREA, TRIANGLE_MAX_AREA))
    
    return all_detections


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main():
    input_video = INPUT_VIDEO_PATH
    final_output = OUTPUT_VIDEO_PATH
    
    # Check video
    cap_test = cv2.VideoCapture(input_video)
    if not cap_test.isOpened():
        print(f"Error: Cannot open video '{input_video}'")
        return
    cap_test.release()
    
    # Get video parameters
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: 
        fps = 30.0
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print("=" * 70)
    print("   TRAFFIC SIGN DETECTION (SINGLE-THREADED VERSION)")
    print("=" * 70)
    print(f"Video: {input_video}")
    print(f"Resolution: {w_orig}x{h_orig} @ {fps:.2f} FPS")
    print(f"Total frames: {total_frames}")
    print(f"Processing limit: {MAX_FRAME_ID} frames")
    print(f"\nCOLOR-SPECIFIC TEMPORAL PARAMETERS:")
    print(f"   Blue:   min={BLUE_MIN_DURATION_SEC}s, gap={BLUE_MAX_GAP_SEC}s, iou={BLUE_IOU_THRESHOLD}")
    print(f"   Red:    min={RED_MIN_DURATION_SEC}s, gap={RED_MAX_GAP_SEC}s, iou={RED_IOU_THRESHOLD}")
    print(f"   Yellow: min={YELLOW_MIN_DURATION_SEC}s, gap={YELLOW_MAX_GAP_SEC}s, iou={YELLOW_IOU_THRESHOLD}")
    print(f"\nAREA PARAMETERS:")
    print(f"   Circle:   min={CIRCLE_MIN_AREA}, max={CIRCLE_MAX_AREA}")
    print(f"   Triangle: min={TRIANGLE_MIN_AREA}, max={TRIANGLE_MAX_AREA}")
    print(f"\nROI SETTINGS (Full Frame %):")
    print(f"   Blue ROI:   x={BLUE_ROI[0]*100:.1f}%-{BLUE_ROI[2]*100:.1f}%, y={BLUE_ROI[1]*100:.1f}%-{BLUE_ROI[3]*100:.1f}%")
    print(f"   Red ROI:    x={RED_ROI[0]*100:.1f}%-{RED_ROI[2]*100:.1f}%, y={RED_ROI[1]*100:.1f}%-{RED_ROI[3]*100:.1f}%")
    print(f"   Yellow ROI: x={YELLOW_ROI[0]*100:.1f}%-{YELLOW_ROI[2]*100:.1f}%, y={YELLOW_ROI[1]*100:.1f}%-{YELLOW_ROI[3]*100:.1f}%")
    print(f"\nDEBUG MODE: {'ENABLED' if DEBUG_MODE else 'DISABLED'}")
    if DEBUG_MODE:
        print(f"   Frame ID overlay: Enabled")
        print(f"   ROI visualization: Enabled")
        print(f"   Detection metrics: Enabled")
    else:
        print(f"   Only bounding boxes will be shown")
    print("=" * 70)
    
    # =========================================================================
    # DETECTION PHASE: SIGN DETECTION AND TEMPORAL TRACKING
    # =========================================================================
    
    print("\nDETECTION PHASE: Detecting traffic signs (single-threaded)...")
    start_detection = time.time()
    
    # Create color-specific parameter dictionary
    color_params = {
        'blue': (BLUE_MIN_DURATION_SEC, BLUE_MAX_GAP_SEC, BLUE_IOU_THRESHOLD),
        'red': (RED_MIN_DURATION_SEC, RED_MAX_GAP_SEC, RED_IOU_THRESHOLD),
        'yellow': (YELLOW_MIN_DURATION_SEC, YELLOW_MAX_GAP_SEC, YELLOW_IOU_THRESHOLD)
    }
    
    temporal_filter = TemporalSignFilter(fps, color_params=color_params)
    
    # Video cropping configuration
    # height_new = int(h_orig * 0.475) means we crop to 47.5% of original height
    # This focuses on the upper portion of the frame where traffic signs typically appear
    height_new = int(h_orig * 0.475)
    width_new = w_orig
    
    # Define ROI parameters in FULL FRAME coordinates
    blue_roi_params = convert_roi_to_pixels(BLUE_ROI, w_orig, h_orig)
    red_roi_params = convert_roi_to_pixels(RED_ROI, w_orig, h_orig)
    yellow_roi_params = convert_roi_to_pixels(YELLOW_ROI, w_orig, h_orig)
    
    roi_params_dict = {
        'blue': blue_roi_params,
        'red': red_roi_params,
        'yellow': yellow_roi_params
    }
    
    # Process frames for detection (up to MAX_FRAME_ID)
    print(f"   Reading and processing frames (up to frame {MAX_FRAME_ID})...")
    cap = cv2.VideoCapture(input_video)
    
    frame_count = 0
    last_progress = 0
    
    while cap.isOpened() and frame_count < MAX_FRAME_ID:
        ret, frame_full = cap.read()
        if not ret:
            break
        
        # Process frame and extract detections
        detections = process_single_frame(frame_full, frame_count, height_new, width_new, roi_params_dict)
        
        # Add detections to temporal filter
        temporal_filter.add_detections(frame_count, detections)
        
        frame_count += 1
        
        # Progress update
        progress = int(frame_count / MAX_FRAME_ID * 100)
        if progress >= last_progress + 10:
            print(f"   Processed {frame_count}/{MAX_FRAME_ID} frames ({progress}%)")
            last_progress = progress
    
    cap.release()
    print(f"   Processed {frame_count} frames for detection")
    
    end_detection = time.time()
    
    # Statistics
    total_tracks, valid_tracks = temporal_filter.get_statistics()
    print(f"\nDETECTION PHASE completed in {end_detection - start_detection:.2f}s")
    print(f"Statistics:")
    print(f"   Total tracks: {total_tracks}")
    print(f"   Valid tracks: {valid_tracks}")
    print(f"   Filtered: {total_tracks - valid_tracks}")
    
    # Build detection cache
    temporal_filter.build_detection_cache()
    
    # =========================================================================
    # RENDERING PHASE: APPLY FILTERS AND GENERATE OUTPUT VIDEO
    # =========================================================================
    
    print(f"\nRENDERING PHASE: Generating output video (single-threaded)...")
    start_rendering = time.time()
    
    # Open video for rendering all frames
    cap = cv2.VideoCapture(input_video)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(final_output, fourcc, fps, (w_orig, h_orig))
    
    if not video_writer.isOpened():
        print(f"Error: Cannot create output file '{final_output}'")
        return
    
    frame_num = 0
    last_progress = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get validated detections from cache
        validated = temporal_filter.get_validated_detections(frame_num)
        
        frame_output = frame.copy()
        
        # Draw debug overlays only if debug mode is enabled
        if DEBUG_MODE:
            # Draw frame ID
            frame_output = draw_frame_id(frame_output, frame_num)
            
            # Draw ROI boxes
            frame_output = draw_roi_boxes(frame_output, roi_params_dict)
        
        # Draw detections (with or without metrics based on debug mode)
        frame_output = draw_detections_with_metrics(frame_output, validated, DEBUG_MODE)
        
        # Write frame to output video
        video_writer.write(frame_output)
        
        frame_num += 1
        
        # Progress update
        progress = int(frame_num / total_frames * 100)
        if progress >= last_progress + 10:
            print(f"   Rendered {frame_num}/{total_frames} frames ({progress}%)")
            last_progress = progress
    
    cap.release()
    video_writer.release()
    
    end_rendering = time.time()
    
    print(f"   Rendering phase completed in {end_rendering - start_rendering:.2f}s")
    print(f"   Rendering speed: {total_frames/(end_rendering - start_rendering):.1f} FPS")
    
    # =========================================================================
    # COMPLETION
    # =========================================================================
    
    total_time = time.time() - start_detection
    
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"Output: {final_output}")
    print(f"Output frames: {total_frames} (full video)")
    print(f"Detection frames: {frame_count} (up to frame {MAX_FRAME_ID})")
    print(f"Total time: {total_time:.2f}s, {total_time/60:.2f} min")
    print(f"Overall speed: {total_frames/total_time:.1f} FPS")
    
    if total_tracks > 0:
        print(f"\nFiltering:")
        print(f"   Retention: {valid_tracks/total_tracks*100:.1f}%")
        
    print("\nProcessing mode:")
    print(f"   Single-threaded (no multiprocessing/multithreading)")
    print(f"   Interpolation + Smoothing")
    print(f"   Detection cache (pre-computed)")
    
    if DEBUG_MODE:
        print("\nDebug features enabled:")
        print(f"   Frame ID overlay")
        print(f"   ROI visualization (Blue, Red, Yellow)")
        print(f"   Detection metrics (Area, Circularity/Solidity)")
    print("=" * 70)


if __name__ == "__main__":
    main()
