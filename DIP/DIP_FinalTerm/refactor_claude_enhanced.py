import cv2 
import numpy as np 
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Dict, Optional
import os

# =============================================================================
# CLASS 1: ENHANCED CONFIGURATION
# =============================================================================
class TrafficSignConfig:
    def __init__(self):
        # --- File paths ---
        self.INPUT_VIDEO_PATH = 'video/video2.mp4'
        self.OUTPUT_VIDEO_PATH = 'video_optimized_enhanced.mp4'
        self.STUDENT_IDS = "523H0164_523H0177_523H0145"
        
        # --- Template paths for feature matching ---
        self.TEMPLATE_DIR = 'templates'  # Thư mục chứa ảnh template biển báo
        self.TEMPLATES = {
            'stop': 'templates/stop_sign.jpg',
            'no_entry': 'templates/no_entry.jpg',
            'speed_limit': 'templates/speed_limit.jpg',
            'warning': 'templates/warning.jpg'
        }
        
        # --- Processing limits ---
        self.MAX_FRAME_ID = 10000
        self.PROGRESS_UPDATE_INTERVAL = 100
        
        # --- Performance settings ---
        self.USE_MULTIPROCESSING = True
        self.NUM_WORKERS = 4
        self.BATCH_SIZE = 50
        self.CACHE_FRAMES = False
        
        # --- Feature matching settings ---
        self.ENABLE_FEATURE_MATCHING = True
        self.FEATURE_MATCH_THRESHOLD = 10  # Minimum good matches
        self.FEATURE_DETECTOR = 'SIFT'  # 'SIFT', 'ORB', or 'AKAZE'
        self.MATCH_RATIO_THRESHOLD = 0.75
        
        # --- Enhanced DIP techniques ---
        self.USE_COLOR_RATIO_ANALYSIS = True
        self.USE_EDGE_DENSITY = True
        self.USE_ASPECT_RATIO_FILTER = True
        self.USE_TEXTURE_ANALYSIS = True
        
        # --- Visualization ---
        self.DEBUG_MODE = True
        self.BOX_COLOR = (0, 255, 0)
        self.SHOW_FEATURE_MATCHES = True
        self.SHOW_COLOR_ANALYSIS = True
        
        # --- Color-specific parameters ---
        self.COLOR_PARAMS = {
            'blue': {
                'hsv_lower': np.array([102, 216, 81]),
                'hsv_upper': np.array([144, 255, 227]),
                'morph_ksize': 7, 'open_iter': 1, 'close_iter': 5,
                'blur_ksize': 7,
                'roi': (0.4, 0.0, 0.7, 0.475),
                'shape_type': 'circle',
                'min_color_ratio': 0.15,  # Tỷ lệ màu tối thiểu
                'expected_aspect_ratio': (0.8, 1.2)  # width/height ratio range
            },
            'red': {
                'hsv_lower': np.array([117, 40, 0]),
                'hsv_upper': np.array([179, 255, 255]),
                'morph_ksize': 2, 'open_iter': 2, 'close_iter': 5,
                'blur_ksize': 5,
                'roi': (0.45, 0.2, 1.0, 0.445),
                'shape_type': 'circle',
                'min_color_ratio': 0.20,
                'expected_aspect_ratio': (0.8, 1.2)
            },
            'yellow': {
                'hsv_lower': np.array([8, 111, 100]),
                'hsv_upper': np.array([18, 255, 255]),
                'morph_ksize': 3, 'open_iter': 1, 'close_iter': 5,
                'blur_ksize': 7,
                'roi': (0.45, 0.2375, 0.8, 0.5),
                'shape_type': 'triangle',
                'min_color_ratio': 0.12,
                'expected_aspect_ratio': (0.7, 1.5)
            }
        }
        
        # --- Image processing ---
        self.CLAHE_CLIP_LIMIT = 3.0
        self.CLAHE_TILE_GRID_SIZE = (8, 8)
        self.SATURATION_BOOST_FACTOR = 1.5
        self.PROCESSING_HEIGHT_PERCENT = 0.475
        
        # --- Shape detection parameters ---
        self.SHAPE_PARAMS = {
            'circle': {
                'min_area': 300, 'max_area': 15000,
                'trust_threshold': 725,
                'small_circularity': 0.87,
                'large_circularity': 0.93
            },
            'triangle': {
                'min_area': 400, 'max_area': 50000,
                'trust_threshold': 1500,
                'min_solidity': 0.75,
                'epsilon_factor': 0.03,
                'max_vertices': 7
            }
        }
        
        # --- Temporal filtering ---
        self.TEMPORAL_PARAMS = {
            'blue': {'min_duration_sec': 2.0, 'max_gap_sec': 0.5, 'iou_threshold': 0.3},
            'red': {'min_duration_sec': 3.0, 'max_gap_sec': 0.5, 'iou_threshold': 0.3},
            'yellow': {'min_duration_sec': 3.0, 'max_gap_sec': 0.5, 'iou_threshold': 0.3}
        }


# =============================================================================
# CLASS 2: FEATURE MATCHING ENGINE
# =============================================================================
class FeatureMatchingEngine:
    """Enhanced feature matching for traffic sign recognition"""
    
    def __init__(self, config: TrafficSignConfig):
        self.config = config
        self.templates = {}
        self.template_keypoints = {}
        self.template_descriptors = {}
        
        # Initialize feature detector
        if config.FEATURE_DETECTOR == 'SIFT':
            self.detector = cv2.SIFT_create()
        elif config.FEATURE_DETECTOR == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=1000)
        elif config.FEATURE_DETECTOR == 'AKAZE':
            self.detector = cv2.AKAZE_create()
        else:
            self.detector = cv2.SIFT_create()
        
        # Initialize matcher
        if config.FEATURE_DETECTOR == 'ORB':
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        self._load_templates()
    
    def _load_templates(self):
        """Load and process template images"""
        for sign_type, path in self.config.TEMPLATES.items():
            if os.path.exists(path):
                template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    self.templates[sign_type] = template
                    kp, des = self.detector.detectAndCompute(template, None)
                    self.template_keypoints[sign_type] = kp
                    self.template_descriptors[sign_type] = des
                    print(f"✓ Loaded template '{sign_type}': {len(kp)} keypoints")
    
    def match_sign(self, roi_image: np.ndarray) -> Tuple[Optional[str], int, float]:
        """
        Match ROI with templates using feature matching
        Returns: (sign_type, num_matches, confidence)
        """
        if roi_image is None or roi_image.size == 0:
            return None, 0, 0.0
        
        # Convert to grayscale if needed
        if len(roi_image.shape) == 3:
            roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi_image
        
        # Detect features in ROI
        kp_roi, des_roi = self.detector.detectAndCompute(roi_gray, None)
        
        if des_roi is None or len(kp_roi) < 5:
            return None, 0, 0.0
        
        best_match = None
        best_num_matches = 0
        best_confidence = 0.0
        
        # Try to match with each template
        for sign_type, des_template in self.template_descriptors.items():
            if des_template is None:
                continue
            
            try:
                matches = self.matcher.knnMatch(des_template, des_roi, k=2)
                
                # Apply ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < self.config.MATCH_RATIO_THRESHOLD * n.distance:
                            good_matches.append(m)
                
                num_good = len(good_matches)
                
                if num_good >= self.config.FEATURE_MATCH_THRESHOLD:
                    # Calculate confidence based on number of matches and average distance
                    avg_distance = np.mean([m.distance for m in good_matches]) if good_matches else 100
                    confidence = num_good / (1 + avg_distance * 0.01)
                    
                    if num_good > best_num_matches:
                        best_match = sign_type
                        best_num_matches = num_good
                        best_confidence = confidence
            
            except Exception as e:
                continue
        
        return best_match, best_num_matches, best_confidence


# =============================================================================
# CLASS 3: ENHANCED DIP ANALYZER
# =============================================================================
class EnhancedDIPAnalyzer:
    """Advanced DIP techniques for traffic sign analysis"""
    
    @staticmethod
    def calculate_color_ratio(roi: np.ndarray, color_params: dict) -> float:
        """
        Tính tỷ lệ diện tích màu cụ thể trong ROI
        Returns: ratio (0.0 to 1.0)
        """
        if roi is None or roi.size == 0:
            return 0.0
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, color_params['hsv_lower'], color_params['hsv_upper'])
        
        total_pixels = roi.shape[0] * roi.shape[1]
        color_pixels = np.count_nonzero(mask)
        
        return color_pixels / total_pixels if total_pixels > 0 else 0.0
    
    @staticmethod
    def calculate_edge_density(roi: np.ndarray) -> float:
        """
        Tính mật độ cạnh trong ROI (edges per pixel)
        Returns: edge density (0.0 to 1.0)
        """
        if roi is None or roi.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        edges = cv2.Canny(gray, 50, 150)
        
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_pixels = np.count_nonzero(edges)
        
        return edge_pixels / total_pixels if total_pixels > 0 else 0.0
    
    @staticmethod
    def calculate_aspect_ratio(bbox: Tuple[int, int, int, int]) -> float:
        """
        Tính tỷ lệ width/height
        Returns: aspect ratio
        """
        x, y, w, h = bbox
        return w / h if h > 0 else 0.0
    
    @staticmethod
    def check_aspect_ratio(aspect_ratio: float, expected_range: Tuple[float, float]) -> bool:
        """
        Kiểm tra xem aspect ratio có nằm trong khoảng mong đợi không
        """
        return expected_range[0] <= aspect_ratio <= expected_range[1]
    
    @staticmethod
    def calculate_texture_features(roi: np.ndarray) -> Dict[str, float]:
        """
        Tính các đặc trưng texture sử dụng Local Binary Pattern
        Returns: dict with texture metrics
        """
        if roi is None or roi.size == 0:
            return {'contrast': 0.0, 'uniformity': 0.0}
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        # Calculate standard deviation as texture measure
        std_dev = np.std(gray)
        mean_val = np.mean(gray)
        
        # Calculate contrast
        contrast = std_dev / (mean_val + 1e-6)
        
        # Calculate uniformity (inverse of entropy)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / (hist.sum() + 1e-6)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        uniformity = 1.0 / (entropy + 1.0)
        
        return {
            'contrast': float(contrast),
            'uniformity': float(uniformity),
            'std_dev': float(std_dev),
            'entropy': float(entropy)
        }
    
    @staticmethod
    def validate_detection(roi: np.ndarray, bbox: Tuple, color: str, 
                          color_params: dict) -> Tuple[bool, Dict]:
        """
        Validation tổng hợp sử dụng nhiều kỹ thuật DIP
        Returns: (is_valid, metrics_dict)
        """
        metrics = {}
        
        # 1. Color ratio analysis
        color_ratio = EnhancedDIPAnalyzer.calculate_color_ratio(roi, color_params)
        metrics['color_ratio'] = color_ratio
        
        min_ratio = color_params.get('min_color_ratio', 0.15)
        if color_ratio < min_ratio:
            return False, metrics
        
        # 2. Aspect ratio check
        aspect_ratio = EnhancedDIPAnalyzer.calculate_aspect_ratio(bbox)
        metrics['aspect_ratio'] = aspect_ratio
        
        expected_range = color_params.get('expected_aspect_ratio', (0.7, 1.5))
        if not EnhancedDIPAnalyzer.check_aspect_ratio(aspect_ratio, expected_range):
            return False, metrics
        
        # 3. Edge density
        edge_density = EnhancedDIPAnalyzer.calculate_edge_density(roi)
        metrics['edge_density'] = edge_density
        
        # Traffic signs typically have high edge density
        if edge_density < 0.05:  # Threshold
            return False, metrics
        
        # 4. Texture analysis
        texture = EnhancedDIPAnalyzer.calculate_texture_features(roi)
        metrics.update(texture)
        
        return True, metrics


# =============================================================================
# CLASS 4: TEMPORAL FILTER (Enhanced)
# =============================================================================
class TemporalSignFilter:
    def __init__(self, fps: float, color_params_dict: Dict):
        self.fps = fps
        self.min_frames = {}
        self.max_gap_frames = {}
        self.iou_thresholds = {}
        
        for color, params in color_params_dict.items():
            self.min_frames[color] = int(params['min_duration_sec'] * fps)
            self.max_gap_frames[color] = int(params['max_gap_sec'] * fps)
            self.iou_thresholds[color] = params['iou_threshold']
        
        self.tracks = defaultdict(list)
        self.next_track_id = 0
        self._validated_cache = {}
        self._cache_built = False
        
    def calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
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
    
    def add_detections(self, frame_num: int, detections: List[Tuple]):
        """Add raw detections with enhanced metrics"""
        self._cache_built = False
        
        for detection_data in detections:
            bbox = detection_data[0]
            color = detection_data[1]
            metrics = detection_data[2] if len(detection_data) > 2 else {}
            sign_type = detection_data[3] if len(detection_data) > 3 else None
            
            iou_threshold = self.iou_thresholds.get(color, 0.3)
            max_gap = self.max_gap_frames.get(color, int(0.5 * self.fps))
            
            best_match_id = None
            best_iou = 0
            
            for track_id, track_data in self.tracks.items():
                if not track_data:
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
                    'metrics': metrics,
                    'sign_type': sign_type
                })
            else:
                self.tracks[self.next_track_id] = [{
                    'frame': frame_num, 
                    'bbox': bbox, 
                    'color': color, 
                    'metrics': metrics,
                    'sign_type': sign_type
                }]
                self.next_track_id += 1
    
    def interpolate_missing_frames(self, track_data: List, color: str) -> List:
        if len(track_data) < 2:
            return track_data
        
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
                    interpolated.append({
                        'frame': current['frame'] + j,
                        'bbox': (int(x1 + (x2 - x1) * alpha),
                                int(y1 + (y2 - y1) * alpha),
                                int(w1 + (w2 - w1) * alpha),
                                int(h1 + (h2 - h1) * alpha)),
                        'color': current['color'],
                        'metrics': current.get('metrics', {}),
                        'sign_type': current.get('sign_type'),
                        'interpolated': True
                    })
        
        interpolated.append(track_data[-1])
        return interpolated
    
    def smooth_bounding_boxes(self, track_data: List, window_size: int = 5) -> List:
        if len(track_data) < window_size:
            return track_data
        
        smoothed = []
        half_window = window_size // 2
        
        for i in range(len(track_data)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(track_data), i + half_window + 1)
            
            window_boxes = [d['bbox'] for d in track_data[start_idx:end_idx]]
            
            avg_x = int(np.mean([b[0] for b in window_boxes]))
            avg_y = int(np.mean([b[1] for b in window_boxes]))
            avg_w = int(np.mean([b[2] for b in window_boxes]))
            avg_h = int(np.mean([b[3] for b in window_boxes]))
            
            smoothed_detection = track_data[i].copy()
            smoothed_detection['bbox'] = (avg_x, avg_y, avg_w, avg_h)
            smoothed.append(smoothed_detection)
        
        return smoothed
    
    def get_validated_tracks(self) -> Dict[int, List]:
        validated = {}
        
        for track_id, track_data in self.tracks.items():
            if not track_data:
                continue
            
            color = track_data[0]['color']
            min_frames_required = self.min_frames.get(color, int(2.0 * self.fps))
            
            if len(track_data) >= min_frames_required:
                interpolated = self.interpolate_missing_frames(track_data, color)
                smoothed = self.smooth_bounding_boxes(interpolated)
                validated[track_id] = smoothed
        
        return validated
    
    def build_frame_cache(self):
        if self._cache_built:
            return
        
        self._validated_cache = {}
        validated_tracks = self.get_validated_tracks()
        
        for track_id, track_data in validated_tracks.items():
            for detection in track_data:
                frame_num = detection['frame']
                if frame_num not in self._validated_cache:
                    self._validated_cache[frame_num] = []
                self._validated_cache[frame_num].append(detection)
        
        self._cache_built = True
    
    def get_validated_detections(self, frame_num: int) -> List:
        if not self._cache_built:
            self.build_frame_cache()
        
        return self._validated_cache.get(frame_num, [])


# =============================================================================
# CLASS 5: ENHANCED DETECTOR
# =============================================================================
class EnhancedTrafficSignDetector:
    """Enhanced detector with feature matching and advanced DIP"""
    
    def __init__(self, config: TrafficSignConfig):
        self.config = config
        self.clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT,
            tileGridSize=config.CLAHE_TILE_GRID_SIZE
        )
        
        # Initialize feature matcher if enabled
        self.feature_matcher = None
        if config.ENABLE_FEATURE_MATCHING:
            self.feature_matcher = FeatureMatchingEngine(config)
        
        self.dip_analyzer = EnhancedDIPAnalyzer()
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing with CLAHE and saturation boost"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        v_clahe = self.clahe.apply(v)
        s_boosted = np.clip(s * self.config.SATURATION_BOOST_FACTOR, 0, 255).astype(np.uint8)
        
        hsv_enhanced = cv2.merge([h, s_boosted, v_clahe])
        return cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    
    def extract_roi(self, frame: np.ndarray, bbox: Tuple) -> np.ndarray:
        """Extract ROI from frame with padding"""
        x, y, w, h = bbox
        h_frame, w_frame = frame.shape[:2]
        
        # Add padding
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_frame, x + w + pad)
        y2 = min(h_frame, y + h + pad)
        
        return frame[y1:y2, x1:x2]
    
    def detect_color_shape(self, frame_bgr: np.ndarray, color: str, 
                          roi_bounds: Tuple) -> List[Tuple]:
        """Detect shapes with enhanced validation"""
        params = self.config.COLOR_PARAMS[color]
        shape_type = params['shape_type']
        shape_params = self.config.SHAPE_PARAMS[shape_type]
        
        # Extract ROI
        x_start = int(frame_bgr.shape[1] * roi_bounds[0])
        y_start = int(frame_bgr.shape[0] * roi_bounds[1])
        x_end = int(frame_bgr.shape[1] * roi_bounds[2])
        y_end = int(frame_bgr.shape[0] * roi_bounds[3])
        
        roi = frame_bgr[y_start:y_end, x_start:x_end]
        
        # Color segmentation
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        blurred = cv2.GaussianBlur(hsv, (params['blur_ksize'], params['blur_ksize']), 0)
        mask = cv2.inRange(blurred, params['hsv_lower'], params['hsv_upper'])
        
        # Morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (params['morph_ksize'], params['morph_ksize']))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 
                               iterations=params['open_iter'])
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 
                               iterations=params['close_iter'])
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < shape_params['min_area'] or area > shape_params['max_area']:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            bbox_global = (x + x_start, y + y_start, w, h)
            
            # Extract ROI for validation
            roi_sign = self.extract_roi(frame_bgr, bbox_global)
            
            # Enhanced validation
            if self.config.USE_COLOR_RATIO_ANALYSIS or self.config.USE_ASPECT_RATIO_FILTER:
                is_valid, metrics = self.dip_analyzer.validate_detection(
                    roi_sign, bbox_global, color, params
                )
                
                if not is_valid:
                    continue
            else:
                metrics = {}
            
            # Shape-specific validation
            if shape_type == 'circle':
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                threshold = (shape_params['large_circularity'] if area > shape_params['trust_threshold']
                           else shape_params['small_circularity'])
                
                if circularity < threshold:
                    continue
                
                metrics['circularity'] = circularity
                
            elif shape_type == 'triangle':
                epsilon = params['epsilon_factor'] * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) > shape_params['max_vertices']:
                    continue
                
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                if solidity < shape_params['min_solidity']:
                    continue
                
                metrics['solidity'] = solidity
                metrics['vertices'] = len(approx)
            
            # Feature matching (if enabled)
            sign_type = None
            match_confidence = 0.0
            
            if self.config.ENABLE_FEATURE_MATCHING and self.feature_matcher:
                sign_type, num_matches, match_confidence = self.feature_matcher.match_sign(roi_sign)
                metrics['feature_matches'] = num_matches
                metrics['match_confidence'] = match_confidence
            
            detections.append((bbox_global, color, metrics, sign_type))
        
        return detections
    
    def process_frame(self, frame: np.ndarray) -> List[Tuple]:
        """Process single frame with all colors"""
        frame_enhanced = self.preprocess_frame(frame)
        
        all_detections = []
        
        for color, params in self.config.COLOR_PARAMS.items():
            detections = self.detect_color_shape(frame_enhanced, color, params['roi'])
            all_detections.extend(detections)
        
        return all_detections


# =============================================================================
# CLASS 6: ENHANCED VISUALIZER
# =============================================================================
class EnhancedVisualizer:
    """Enhanced visualization with detailed metrics"""
    
    def __init__(self, config: TrafficSignConfig):
        self.config = config
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.thickness = 1
    
    def draw_enhanced_bbox(self, frame: np.ndarray, detection: dict) -> np.ndarray:
        """Draw bbox with enhanced information"""
        bbox = detection['bbox']
        color = detection['color']
        metrics = detection.get('metrics', {})
        sign_type = detection.get('sign_type')
        
        x, y, w, h = bbox
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.config.BOX_COLOR, 2)
        
        # Prepare label
        label_parts = [color.upper()]
        
        if sign_type:
            label_parts.append(f"[{sign_type}]")
        
        # Add key metrics
        if self.config.SHOW_COLOR_ANALYSIS:
            if 'color_ratio' in metrics:
                label_parts.append(f"CR:{metrics['color_ratio']:.2f}")
            if 'circularity' in metrics:
                label_parts.append(f"Cir:{metrics['circularity']:.2f}")
            if 'solidity' in metrics:
                label_parts.append(f"Sol:{metrics['solidity']:.2f}")
        
        if self.config.SHOW_FEATURE_MATCHES:
            if 'feature_matches' in metrics:
                label_parts.append(f"FM:{metrics['feature_matches']}")
        
        label = " ".join(label_parts)
        
        # Draw label background
        (label_w, label_h), _ = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)
        cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w + 10, y), 
                     self.config.BOX_COLOR, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x + 5, y - 5), self.font, self.font_scale, 
                   (0, 0, 0), self.thickness)
        
        return frame
    
    def draw_roi_zones(self, frame: np.ndarray, roi_map: Dict) -> np.ndarray:
        """Draw ROI zones"""
        overlay = frame.copy()
        
        colors_vis = {'blue': (255, 100, 0), 'red': (0, 100, 255), 'yellow': (0, 200, 200)}
        
        for color, (x1, y1, x2, y2) in roi_map.items():
            cv2.rectangle(overlay, (x1, y1), (x2, y2), colors_vis[color], 2)
            cv2.putText(overlay, f"{color.upper()} ROI", (x1 + 10, y1 + 25),
                       self.font, 0.7, colors_vis[color], 2)
        
        return cv2.addWeighted(frame, 0.9, overlay, 0.1, 0)
    
    def draw_frame_info(self, frame: np.ndarray, frame_id: int, 
                       num_detections: int) -> np.ndarray:
        """Draw frame information"""
        info_text = f"Frame: {frame_id} | Detections: {num_detections}"
        
        cv2.rectangle(frame, (10, 10), (400, 40), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (15, 30), self.font, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def draw_all(self, frame: np.ndarray, frame_id: int, 
                detections: List[dict], roi_map: Dict) -> np.ndarray:
        """Draw all visualizations"""
        if self.config.DEBUG_MODE:
            frame = self.draw_roi_zones(frame, roi_map)
            frame = self.draw_frame_info(frame, frame_id, len(detections))
        
        for detection in detections:
            frame = self.draw_enhanced_bbox(frame, detection)
        
        return frame


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================
def main():
    print("\n" + "=" * 70)
    print("ENHANCED TRAFFIC SIGN DETECTION SYSTEM")
    print("Features: Color Analysis + Feature Matching + Advanced DIP")
    print("=" * 70 + "\n")
    
    # Initialize
    config = TrafficSignConfig()
    
    # Check template directory
    if config.ENABLE_FEATURE_MATCHING:
        if not os.path.exists(config.TEMPLATE_DIR):
            print(f"⚠ Warning: Template directory '{config.TEMPLATE_DIR}' not found")
            print("   Feature matching will be disabled")
            config.ENABLE_FEATURE_MATCHING = False
    
    try:
        start_total = time.time()
        
        # Open video
        cap = cv2.VideoCapture(config.INPUT_VIDEO_PATH)
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video '{config.INPUT_VIDEO_PATH}'")
            return
        
        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), config.MAX_FRAME_ID)
        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Video: {config.INPUT_VIDEO_PATH}")
        print(f"   Resolution: {w_orig}x{h_orig}")
        print(f"   FPS: {fps:.2f}")
        print(f"   Total frames: {total_frames}")
        
        # Calculate ROI pixel coordinates
        roi_pixel_map = {}
        for color, params in config.COLOR_PARAMS.items():
            x1 = int(w_orig * params['roi'][0])
            y1 = int(h_orig * params['roi'][1])
            x2 = int(w_orig * params['roi'][2])
            y2 = int(h_orig * params['roi'][3])
            roi_pixel_map[color] = (x1, y1, x2, y2)
        
        # Initialize components
        detector = EnhancedTrafficSignDetector(config)
        temporal_filter = TemporalSignFilter(fps, config.TEMPORAL_PARAMS)
        visualizer = EnhancedVisualizer(config)
        
        print(f"\n{'='*70}")
        print("PHASE 1: DETECTION")
        print(f"{'='*70}")
        
        start_detection = time.time()
        frame_count = 0
        
        # Detection loop
        while cap.isOpened() and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            detections = detector.process_frame(frame)
            temporal_filter.add_detections(frame_count, detections)
            
            frame_count += 1
            
            if frame_count % config.PROGRESS_UPDATE_INTERVAL == 0:
                progress = int(frame_count / total_frames * 100)
                print(f"   Processed {frame_count}/{total_frames} frames ({progress}%) | "
                      f"Detections: {len(detections)}")
        
        detection_time = time.time() - start_detection
        
        print(f"\n✓ Detection completed in {detection_time:.2f}s")
        print(f"  Average speed: {frame_count/detection_time:.1f} FPS")
        
        # Build temporal cache
        print("\n🔄 Building temporal filter cache...")
        temporal_filter.build_frame_cache()
        validated_tracks = temporal_filter.get_validated_tracks()
        print(f"✓ Found {len(validated_tracks)} validated tracks")
        
        # Rendering phase
        print(f"\n{'='*70}")
        print("PHASE 2: RENDERING")
        print(f"{'='*70}")
        
        start_render = time.time()
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(config.OUTPUT_VIDEO_PATH, fourcc, fps, 
                                       (w_orig, h_orig))
        
        if not video_writer.isOpened():
            print(f"❌ Error: Cannot create output '{config.OUTPUT_VIDEO_PATH}'")
            cap.release()
            return
        
        frame_count_render = 0
        frames_with_detections = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            validated_detections = temporal_filter.get_validated_detections(frame_count_render)
            
            if validated_detections:
                frames_with_detections += 1
            
            frame_output = visualizer.draw_all(frame, frame_count_render, 
                                              validated_detections, roi_pixel_map)
            
            video_writer.write(frame_output)
            
            frame_count_render += 1
            
            if frame_count_render % config.PROGRESS_UPDATE_INTERVAL == 0:
                progress = int(frame_count_render / total_frames * 100)
                print(f"   Rendered {frame_count_render}/{total_frames} frames ({progress}%)")
        
        render_time = time.time() - start_render
        
        print(f"\n✓ Rendering completed in {render_time:.2f}s")
        print(f"  Speed: {frame_count_render/render_time:.1f} FPS")
        print(f"  Frames with detections: {frames_with_detections}/{frame_count_render}")
        
        cap.release()
        video_writer.release()
        
        # Summary
        total_time = time.time() - start_total
        
        print("\n" + "=" * 70)
        print("✓ PROCESSING COMPLETE!")
        print("=" * 70)
        print(f"Output: {config.OUTPUT_VIDEO_PATH}")
        print(f"\nPerformance:")
        print(f"   Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"   Detection: {detection_time:.2f}s ({detection_time/total_time*100:.1f}%)")
        print(f"   Rendering: {render_time:.2f}s ({render_time/total_time*100:.1f}%)")
        print(f"   Overall speed: {frame_count_render/total_time:.1f} FPS")
        
        print(f"\n✨ Enhanced Features:")
        print(f"   ✓ Color ratio analysis")
        print(f"   ✓ Aspect ratio filtering")
        print(f"   ✓ Edge density calculation")
        print(f"   ✓ Texture analysis")
        if config.ENABLE_FEATURE_MATCHING:
            print(f"   ✓ Feature matching (SIFT/ORB)")
        print(f"   ✓ Temporal filtering")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            cap.release()
            video_writer.release()
        except:
            pass


if __name__ == "__main__":
    main()
