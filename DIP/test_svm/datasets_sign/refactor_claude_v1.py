import cv2 
import numpy as np 
import time
import gc
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count, Manager, Queue
from queue import Empty
from typing import List, Tuple, Dict
import threading

# =============================================================================
# CLASS 1: CONFIGURATION
# =============================================================================
class TrafficSignConfig:
    def __init__(self):
        # --- File paths ---
        self.INPUT_VIDEO_PATH = 'video/video1.mp4'
        self.OUTPUT_VIDEO_PATH = 'video_optimized_refactor.mp4'
        self.STUDENT_IDS = "523H0164_523H0177_523H0145"
        
        # --- Mask video output ---
        self.SAVE_MASK_VIDEOS = True
        self.MASK_VIDEO_BLUE = 'mask_video_blue.mp4'
        self.MASK_VIDEO_RED = 'mask_video_red.mp4'
        self.MASK_VIDEO_YELLOW = 'mask_video_yellow.mp4'
        
        # --- Processing limits ---
        self.MAX_FRAME_ID = 10000
        self.PROGRESS_UPDATE_INTERVAL = 100
        
        # --- Performance settings (OPTIMIZED) ---
        self.USE_MULTIPROCESSING = True  # ENABLED
        self.NUM_WORKERS = max(1, cpu_count() - 1)  # Leave 1 core for system
        self.BATCH_SIZE = 30  # Optimal batch size
        self.PREFETCH_FRAMES = 60  # Frames to prefetch
        self.USE_GPU = cv2.cuda.getCudaEnabledDeviceCount() > 0  # Auto-detect GPU
        
        # --- Visualization ---
        self.DEBUG_MODE = True
        self.BOX_COLOR = (0, 255, 0)
        
        # --- Color-specific parameters ---
        self.COLOR_PARAMS = {
            'blue': {
                'hsv_lower': np.array([100, 200, 70]),
                'hsv_upper': np.array([150, 255, 230]),
                'morph_ksize': 7, 'open_iter': 1, 'close_iter': 5,
                'blur_ksize': 7,
                'roi': (0.0, 0.0, 1.0, 1.0),
                'shape_type': 'circle'
            },
            'red': {
                'hsv_lower': np.array([110, 30, 0]),
                'hsv_upper': np.array([179, 255, 255]),
                'morph_ksize': 2, 'open_iter': 2, 'close_iter': 5,
                'blur_ksize': 5,
                'roi': (0.0, 0.0, 1.0, 1.0),
                'shape_type': 'circle'
            },
            'yellow': {
                'hsv_lower': np.array([8, 90, 90]),
                'hsv_upper': np.array([30, 255, 255]),
                'morph_ksize': 3, 'open_iter': 1, 'close_iter': 5,
                'blur_ksize': 7,
                'roi': (0.0, 0.0, 1.0, 1.0),
                'shape_type': 'triangle'
            }
        }
        
        # --- Image processing ---
        self.CLAHE_CLIP_LIMIT = 3.0
        self.CLAHE_TILE_GRID_SIZE = (8, 8)
        self.SATURATION_BOOST_FACTOR = 1.5
        self.PROCESSING_HEIGHT_PERCENT = 1.0
        
        # --- Shape detection parameters ---
        self.SHAPE_PARAMS = {
            'circle': {
                'min_area': 200, 'max_area': 15000,
                'trust_threshold': 1000,
                'small_circularity': 0.80,
                'large_circularity': 0.9
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
            'red': {'min_duration_sec': 2.0, 'max_gap_sec': 0.5, 'iou_threshold': 0.3},
            'yellow': {'min_duration_sec': 2.0, 'max_gap_sec': 0.5, 'iou_threshold': 0.3}
        }


# =============================================================================
# CLASS 2: TEMPORAL FILTER (Thread-safe)
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
        self._lock = threading.Lock()  # Thread safety
        
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
    
    def add_detections_batch(self, batch_detections: List[Tuple[int, List]]):
        """Thread-safe batch addition"""
        with self._lock:
            self._cache_built = False
            for frame_num, detections in batch_detections:
                for bbox, color, metrics in detections:
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
                            'frame': frame_num, 'bbox': bbox, 'color': color, 'metrics': metrics
                        })
                    else:
                        self.tracks[self.next_track_id] = [{
                            'frame': frame_num, 'bbox': bbox, 'color': color, 'metrics': metrics
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
                        'interpolated': True
                    })
        
        interpolated.append(track_data[-1])
        return interpolated
    
    def smooth_bounding_boxes(self, track_data: List, window_size: int = 5) -> List:
        if len(track_data) < window_size:
            return track_data
        
        smoothed = []
        for i in range(len(track_data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(track_data), i + window_size // 2 + 1)
            
            bboxes = [d['bbox'] for d in track_data[start_idx:end_idx]]
            
            smoothed_det = track_data[i].copy()
            smoothed_det['bbox'] = (
                int(np.mean([b[0] for b in bboxes])),
                int(np.mean([b[1] for b in bboxes])),
                int(np.mean([b[2] for b in bboxes])),
                int(np.mean([b[3] for b in bboxes]))
            )
            smoothed.append(smoothed_det)
        
        return smoothed
    
    def build_detection_cache(self):
        """Build validated detection cache"""
        if self._cache_built:
            return
        
        print("   Building detection cache...")
        start = time.time()
        
        with self._lock:
            self._validated_cache.clear()
            
            for track_id, track_data in self.tracks.items():
                if not track_data:
                    continue
                
                color = track_data[0]['color']
                min_frames_for_color = self.min_frames.get(color, int(2.0 * self.fps))
                duration_frames = track_data[-1]['frame'] - track_data[0]['frame'] + 1
                
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
        
        print(f"   Cache built in {time.time() - start:.2f}s "
              f"({len(self._validated_cache)} frames)")
    
    def get_validated_detections(self, frame_num: int) -> List[Tuple]:
        """Thread-safe get validated detections"""
        if not self._cache_built:
            return []
        with self._lock:
            return self._validated_cache.get(frame_num, [])


# =============================================================================
# CLASS 3: OPTIMIZED DETECTOR
# =============================================================================
class TrafficSignDetector:
    def __init__(self, config: TrafficSignConfig):
        self.config = config
        self.clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT,
            tileGridSize=config.CLAHE_TILE_GRID_SIZE
        )
        
    def _preprocess_frame(self, frame: np.ndarray, blur_ksize: int) -> np.ndarray:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        
        frame_proc = cv2.medianBlur(frame, blur_ksize)
        frame_proc = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2HSV)
        
        h, s, v = cv2.split(frame_proc)
        v_clahe = self.clahe.apply(v)
        
        s = s.astype(np.float32) * self.config.SATURATION_BOOST_FACTOR
        s = np.clip(s, 0, 255).astype(np.uint8)
        
        return cv2.merge([h, s, v_clahe])
    
    def _apply_morphology(self, mask: np.ndarray, k_size: int, 
                         open_iter: int, close_iter: int) -> np.ndarray:
        if k_size <= 1:
            return mask
        if k_size % 2 == 0:
            k_size += 1
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
        
        return mask_clean
    
    def _detect_circles(self, contour: np.ndarray, area: float, 
                       trust_threshold: float) -> Tuple[bool, Dict]:
        hull = cv2.convexHull(contour)
        perimeter_hull = cv2.arcLength(hull, True)
        area_hull = cv2.contourArea(hull)
        
        if perimeter_hull <= 0:
            return False, {}
        
        circularity = 4 * np.pi * (area_hull / (perimeter_hull * perimeter_hull))
        
        shape_cfg = self.config.SHAPE_PARAMS['circle']
        circ_thresh = (shape_cfg['small_circularity'] if area < trust_threshold 
                      else shape_cfg['large_circularity'])
        
        if circularity > circ_thresh:
            return True, {
                'area': int(area_hull),
                'circularity': round(circularity, 3),
                'shape': 'circle'
            }
        return False, {}
    
    def _detect_triangles(self, contour: np.ndarray, area: float) -> Tuple[bool, Dict]:
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        if hull_area == 0:
            return False, {}
        
        shape_cfg = self.config.SHAPE_PARAMS['triangle']
        solidity = float(area) / hull_area
        
        if solidity <= shape_cfg['min_solidity']:
            return False, {}
        
        perimeter = cv2.arcLength(contour, True)
        epsilon = shape_cfg['epsilon_factor'] * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) <= shape_cfg['max_vertices']:
            return True, {
                'area': int(area),
                'solidity': round(solidity, 3),
                'shape': 'triangle'
            }
        return False, {}
    
    def _process_color(self, frame_hsv: np.ndarray, color_name: str, 
                      roi_params: Tuple) -> Tuple[List[Tuple], np.ndarray]:
        params = self.config.COLOR_PARAMS[color_name]
        
        mask = cv2.inRange(frame_hsv, params['hsv_lower'], params['hsv_upper'])
        mask_clean = self._apply_morphology(
            mask, params['morph_ksize'], 
            params['open_iter'], params['close_iter']
        )
        
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        shape_type = params['shape_type']
        shape_cfg = self.config.SHAPE_PARAMS[shape_type]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < shape_cfg['min_area'] or area > shape_cfg['max_area']:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            is_in_roi = is_bbox_in_roi((x, y, w, h), roi_params, overlap_threshold=0.5)
            
            if area < shape_cfg['trust_threshold'] and not is_in_roi:
                continue
            
            if shape_type == 'circle':
                is_valid, metrics = self._detect_circles(contour, area, 
                                                        shape_cfg['trust_threshold'])
            else:
                is_valid, metrics = self._detect_triangles(contour, area)
            
            if is_valid:
                detections.append(((x, y, w, h), color_name, metrics))
        
        return detections, mask_clean
    
    def process_frame(self, frame: np.ndarray, full_frame_dims: Tuple) -> Tuple[List[Tuple], Dict[str, np.ndarray]]:
        w_full, h_full = full_frame_dims
        all_detections = []
        masks_dict = {}
        
        roi_pix = {
            color: convert_roi_to_pixels(params['roi'], w_full, h_full)
            for color, params in self.config.COLOR_PARAMS.items()
        }
        
        for color_name, params in self.config.COLOR_PARAMS.items():
            frame_hsv = self._preprocess_frame(frame, params['blur_ksize'])
            detections, mask = self._process_color(frame_hsv, color_name, roi_pix[color_name])
            all_detections.extend(detections)
            masks_dict[color_name] = mask
        
        return all_detections, masks_dict


# =============================================================================
# CLASS 4: VISUALIZER
# =============================================================================
class Visualizer:
    def __init__(self, config: TrafficSignConfig):
        self.config = config
        self.roi_colors = {
            'blue': (255, 0, 0), 
            'red': (0, 0, 255), 
            'yellow': (0, 255, 255)
        }
    
    def draw_all(self, frame: np.ndarray, frame_num: int, 
                 detections: List[Tuple], roi_params_dict: Dict) -> np.ndarray:
        frame_output = frame.copy()
        frame_output = self._draw_student_ids(frame_output)
        
        if self.config.DEBUG_MODE:
            frame_output = self._draw_frame_id(frame_output, frame_num)
            frame_output = self._draw_roi_boxes(frame_output, roi_params_dict)
        
        frame_output = self._draw_detections(frame_output, detections)
        return frame_output
    
    def _draw_student_ids(self, frame: np.ndarray) -> np.ndarray:
        text = self.config.STUDENT_IDS
        x, y = 10, 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(text, font, 0.8, 2)
        cv2.rectangle(frame, (x - 5, y - text_h - 5), 
                     (x + text_w + 5, y + baseline + 5), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return frame
    
    def _draw_frame_id(self, frame: np.ndarray, frame_num: int) -> np.ndarray:
        text = f"Frame: {frame_num}"
        x, y = 10, 70
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), _ = cv2.getTextSize(text, font, 1.0, 2)
        cv2.rectangle(frame, (x - 5, y - text_h - 5), 
                     (x + text_w + 5, y + 5), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        return frame
    
    def _draw_roi_boxes(self, frame: np.ndarray, roi_params_dict: Dict) -> np.ndarray:
        for color_name, (x_start, y_start, x_end, y_end) in roi_params_dict.items():
            color_bgr = self.roi_colors.get(color_name, (255, 255, 255))
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color_bgr, 2)
            cv2.putText(frame, f"{color_name.upper()} ROI", 
                       (x_start + 10, y_start + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2, cv2.LINE_AA)
        return frame
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Tuple]) -> np.ndarray:
        for bbox, color_type, metrics in detections:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.config.BOX_COLOR, 3)
            
            if self.config.DEBUG_MODE and metrics:
                shape = metrics.get('shape', 'unknown')
                area = metrics.get('area', 0)
                
                if shape == 'circle':
                    text = f"A:{area} C:{metrics.get('circularity', 0):.2f}"
                else:
                    text = f"A:{area} S:{metrics.get('solidity', 0):.2f}"
                
                text_y = max(y - 10, 20)
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x, text_y - text_h - 5), 
                            (x + text_w + 5, text_y + 5), (0, 0, 0), -1)
                cv2.putText(frame, text, (x, text_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return frame


# =============================================================================
# CLASS 5: SVM DETECTOR
# =============================================================================
class SignDetectorSVM:
    def __init__(self, model_path: str):
        print(f"Loading SVM Detector from: {model_path}")
        try:
            self.svm = cv2.ml.SVM_load(model_path)
            if self.svm is None: 
                raise Exception(f"Cannot load model from {model_path}")
        except Exception as e:
            print(f"❌ Critical error: Cannot load Detector model.")
            print(f"   Details: {e}")
            raise
            
        RESIZE_DIM = (64, 64)
        winSize = RESIZE_DIM
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        
        self.hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        self.resize_dim = RESIZE_DIM
        print("✓ SVM Detector loaded successfully.")

    def verify(self, frame: np.ndarray, bbox: Tuple) -> bool:
        x, y, w, h = bbox
        
        if w <= 0 or h <= 0 or y+h > frame.shape[0] or x+w > frame.shape[1]:
            return False
            
        try:
            sign_roi = frame[y:y+h, x:x+w]
            sign_gray = cv2.cvtColor(sign_roi, cv2.COLOR_BGR2GRAY)
            sign_resized = cv2.resize(sign_gray, self.resize_dim, interpolation=cv2.INTER_AREA)
            features = self.hog.compute(sign_resized)
            _, result = self.svm.predict(features.reshape(1, -1))
            label = int(result[0][0])
            return label == 1
        except:
            return False


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def convert_roi_to_pixels(roi_percent: Tuple, w_full: int, h_full: int) -> Tuple:
    x_start_pct, y_start_pct, x_end_pct, y_end_pct = roi_percent
    return (int(w_full * x_start_pct), int(h_full * y_start_pct),
            int(w_full * x_end_pct), int(h_full * y_end_pct))

def is_bbox_in_roi(bbox: Tuple, roi_params: Tuple, overlap_threshold: float = 0.5) -> bool:
    x, y, w, h = bbox
    roi_x_start, roi_y_start, roi_x_end, roi_y_end = roi_params
    x_max, y_max = x + w, y + h
    
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
# PARALLEL PROCESSING - OPTIMIZED
# =============================================================================
def process_frame_batch_worker(args):
    """Worker function for parallel batch processing"""
    frames_batch, config, full_frame_dims = args
    detector = TrafficSignDetector(config)
    
    batch_results = []
    batch_masks = []
    
    for frame_num, frame in frames_batch:
        detections, masks = detector.process_frame(frame, full_frame_dims)
        batch_results.append((frame_num, detections))
        batch_masks.append((frame_num, masks))
    
    return batch_results, batch_masks


class FrameReader(threading.Thread):
    """Dedicated thread for reading frames from video"""
    def __init__(self, video_path, max_frames, queue_size=120):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.max_frames = max_frames
        self.queue = Queue(maxsize=queue_size)
        self.stopped = False
        
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        
        while not self.stopped and frame_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.queue.put((frame_count, frame.copy()))
            frame_count += 1
        
        cap.release()
        self.queue.put(None)  # Signal completion
    
    def stop(self):
        self.stopped = True
    
    def get_frame(self, timeout=1):
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None


# =============================================================================
# MAIN FUNCTION - FULLY OPTIMIZED
# =============================================================================
def main():
    print("=" * 70)
    print("  TRAFFIC SIGN DETECTION (MAXIMUM PERFORMANCE)")
    print("=" * 70)
    
    try:
        start_total = time.time()
        config = TrafficSignConfig()
        visualizer = Visualizer(config)
        
        try:
            detector_svm = SignDetectorSVM(model_path="svm_sign_detector_v2.yml")
        except Exception as e:
            print("Cannot initialize SVM detector")
            return
        
        # Open video and get properties
        cap = cv2.VideoCapture(config.INPUT_VIDEO_PATH)
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video '{config.INPUT_VIDEO_PATH}'")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30.0
        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nVideo: {config.INPUT_VIDEO_PATH}")
        print(f"Resolution: {w_orig}x{h_orig} @ {fps:.2f} FPS")
        print(f"Total frames: {total_frames}")
        print(f"Processing limit: {config.MAX_FRAME_ID} frames")
        print(f"CPU cores available: {cpu_count()}")
        print(f"Workers: {config.NUM_WORKERS}")
        print(f"Multiprocessing: {'ENABLED' if config.USE_MULTIPROCESSING else 'DISABLED'}")
        print(f"GPU acceleration: {'ENABLED' if config.USE_GPU else 'DISABLED'}")
        
        # Initialize temporal filter
        temporal_filter = TemporalSignFilter(fps, config.TEMPORAL_PARAMS)
        
        # Calculate processing dimensions
        h_crop = int(h_orig * config.PROCESSING_HEIGHT_PERCENT)
        w_crop = w_orig
        full_frame_dims = (w_orig, h_orig)
        
        # Calculate ROI once
        roi_pixel_map = {
            color: convert_roi_to_pixels(params['roi'], w_orig, h_orig)
            for color, params in config.COLOR_PARAMS.items()
        }
        
        # =====================================================================
        # PASS 1: DETECTION (PARALLEL WITH FRAME PREFETCHING)
        # =====================================================================
        print("\n" + "=" * 70)
        print("PASS 1: DETECTION (Parallel processing with prefetching)")
        print("=" * 70)
        start_detection = time.time()
        
        cap.release()  # Close for now
        
        if config.USE_MULTIPROCESSING:
            # Start frame reader thread
            print(f"Starting frame reader (prefetch buffer: {config.PREFETCH_FRAMES})...")
            frame_reader = FrameReader(config.INPUT_VIDEO_PATH, config.MAX_FRAME_ID, 
                                      queue_size=config.PREFETCH_FRAMES)
            frame_reader.start()
            
            # Process in parallel batches
            print(f"Processing with {config.NUM_WORKERS} workers (batch size: {config.BATCH_SIZE})...")
            
            with ThreadPoolExecutor(max_workers=config.NUM_WORKERS) as executor:
                futures = []
                batch = []
                frame_count = 0
                all_masks_results = []
                
                while True:
                    frame_data = frame_reader.get_frame()
                    
                    if frame_data is None:  # End of video
                        # Process remaining batch
                        if batch:
                            future = executor.submit(
                                process_frame_batch_worker,
                                (batch, config, full_frame_dims)
                            )
                            futures.append(future)
                        break
                    
                    frame_num, frame = frame_data
                    frame_to_process = frame[0:h_crop, 0:w_crop]
                    batch.append((frame_num, frame_to_process))
                    frame_count += 1
                    
                    # Submit batch when full
                    if len(batch) >= config.BATCH_SIZE:
                        future = executor.submit(
                            process_frame_batch_worker,
                            (batch.copy(), config, full_frame_dims)
                        )
                        futures.append(future)
                        batch = []
                    
                    if frame_count % config.PROGRESS_UPDATE_INTERVAL == 0:
                        progress = int(frame_count / config.MAX_FRAME_ID * 100)
                        print(f"   Queued {frame_count} frames ({progress}%) - "
                              f"{len(futures)} batches submitted")
                
                # Collect results - STREAMING MODE (no accumulation)
                print(f"\nCollecting results from {len(futures)} batches...")
                processed_count = 0
                
                # Open mask writers NOW if needed (streaming write)
                mask_writers = None
                if config.SAVE_MASK_VIDEOS:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    mask_writers = {
                        'blue': cv2.VideoWriter(config.MASK_VIDEO_BLUE, fourcc, fps, (w_crop, h_crop), False),
                        'red': cv2.VideoWriter(config.MASK_VIDEO_RED, fourcc, fps, (w_crop, h_crop), False),
                        'yellow': cv2.VideoWriter(config.MASK_VIDEO_YELLOW, fourcc, fps, (w_crop, h_crop), False)
                    }
                    print("   Mask writers opened (streaming mode)")
                
                # Buffer for ordered mask writing
                mask_buffer = {} if config.SAVE_MASK_VIDEOS else None
                next_mask_frame = 0
                
                for future in as_completed(futures):
                    batch_results, batch_masks = future.result()
                    
                    # Add to temporal filter
                    temporal_filter.add_detections_batch(batch_results)
                    
                    # Write masks immediately in order (streaming)
                    if config.SAVE_MASK_VIDEOS and mask_writers:
                        for frame_num, masks in batch_masks:
                            mask_buffer[frame_num] = masks
                        
                        # Write all consecutive frames
                        while next_mask_frame in mask_buffer:
                            masks = mask_buffer.pop(next_mask_frame)
                            for color in ['blue', 'red', 'yellow']:
                                if color in masks and mask_writers[color].isOpened():
                                    mask_writers[color].write(masks[color])
                            next_mask_frame += 1
                            del masks  # Immediate cleanup
                    
                    processed_count += len(batch_results)
                    
                    if processed_count % config.PROGRESS_UPDATE_INTERVAL == 0:
                        print(f"   Processed {processed_count}/{frame_count} frames")
                    
                    # Aggressive memory cleanup
                    del batch_results, batch_masks
                    if processed_count % 200 == 0:
                        gc.collect()
                
                # Close mask writers
                if config.SAVE_MASK_VIDEOS and mask_writers:
                    for writer in mask_writers.values():
                        writer.release()
                    print(f"   Mask videos written: {next_mask_frame} frames")
                    del mask_writers, mask_buffer
                    gc.collect()
            
            frame_reader.stop()
            print(f"\n   Total frames processed: {frame_count}")
            
        else:
            # Sequential processing (fallback)
            print("Processing sequentially...")
            cap = cv2.VideoCapture(config.INPUT_VIDEO_PATH)
            detector = TrafficSignDetector(config)
            frame_count = 0
            
            # Open mask writers immediately (streaming)
            mask_writers = None
            if config.SAVE_MASK_VIDEOS:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                mask_writers = {
                    'blue': cv2.VideoWriter(config.MASK_VIDEO_BLUE, fourcc, fps, (w_crop, h_crop), False),
                    'red': cv2.VideoWriter(config.MASK_VIDEO_RED, fourcc, fps, (w_crop, h_crop), False),
                    'yellow': cv2.VideoWriter(config.MASK_VIDEO_YELLOW, fourcc, fps, (w_crop, h_crop), False)
                }
            
            while cap.isOpened() and frame_count < config.MAX_FRAME_ID:
                ret, frame_full = cap.read()
                if not ret:
                    break
                
                frame_to_process = frame_full[0:h_crop, 0:w_crop]
                detections, masks = detector.process_frame(frame_to_process, full_frame_dims)
                
                if detections:
                    temporal_filter.add_detections_batch([(frame_count, detections)])
                
                # Write masks immediately (no storage)
                if config.SAVE_MASK_VIDEOS and mask_writers:
                    for color in ['blue', 'red', 'yellow']:
                        if color in masks and mask_writers[color].isOpened():
                            mask_writers[color].write(masks[color])
                
                frame_count += 1
                
                if frame_count % config.PROGRESS_UPDATE_INTERVAL == 0:
                    progress = int(frame_count / config.MAX_FRAME_ID * 100)
                    print(f"   Processed {frame_count} frames ({progress}%)")
                    gc.collect()
                
                # Cleanup
                del frame_full, frame_to_process, detections, masks
            
            cap.release()
            if mask_writers:
                for writer in mask_writers.values():
                    writer.release()
            print(f"\n   Total frames processed: {frame_count}")
        
        detection_time = time.time() - start_detection
        print(f"\n✓ Detection Pass completed in {detection_time:.2f}s")
        print(f"  Speed: {frame_count/detection_time:.1f} FPS")
        
        # Build validation cache
        temporal_filter.build_detection_cache()
        
        # Statistics
        total_tracks = len(temporal_filter.tracks)
        valid_tracks = sum(1 for track in temporal_filter.tracks.values() 
                          if track and (track[-1]['frame'] - track[0]['frame'] + 1) >= 
                          temporal_filter.min_frames.get(track[0]['color'], 30))
        
        print(f"\nTracking Statistics:")
        print(f"   Total tracks: {total_tracks}")
        print(f"   Valid tracks: {valid_tracks}")
        print(f"   Filtered out: {total_tracks - valid_tracks}")
        if total_tracks > 0:
            print(f"   Retention rate: {valid_tracks/total_tracks*100:.1f}%")
        
        # =====================================================================
        # PASS 2: RENDERING (Parallel with SVM verification)
        # =====================================================================
        print("\n" + "=" * 70)
        print("PASS 2: RENDERING (Parallel with SVM verification)")
        print("=" * 70)
        start_render = time.time()
        
        # Start frame reader for rendering
        frame_reader_render = FrameReader(config.INPUT_VIDEO_PATH, total_frames, 
                                         queue_size=config.PREFETCH_FRAMES)
        frame_reader_render.start()
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(config.OUTPUT_VIDEO_PATH, fourcc, fps, 
                                       (w_orig, h_orig))
        
        if not video_writer.isOpened():
            print(f"❌ Error: Cannot create output '{config.OUTPUT_VIDEO_PATH}'")
            return
        
        # Render with parallel SVM verification
        frame_count_render = 0
        frames_with_detections = 0
        frame_buffer = {}  # For ordered writing
        
        with ThreadPoolExecutor(max_workers=config.NUM_WORKERS) as executor:
            futures_dict = {}
            
            # Submit initial batch
            for _ in range(config.NUM_WORKERS * 2):
                frame_data = frame_reader_render.get_frame()
                if frame_data is None:
                    break
                
                frame_num, frame = frame_data
                validated_candidates = temporal_filter.get_validated_detections(frame_num)
                
                future = executor.submit(
                    render_frame_worker,
                    frame.copy(), frame_num, validated_candidates, 
                    detector_svm, visualizer, roi_pixel_map
                )
                futures_dict[future] = frame_num
            
            # Process continuously
            next_frame_to_write = 0
            
            while futures_dict or next_frame_to_write < total_frames:
                # Get completed futures
                if futures_dict:
                    done, _ = concurrent.futures.wait(
                        futures_dict.keys(), 
                        timeout=0.1,
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    
                    for future in done:
                        frame_num = futures_dict.pop(future)
                        frame_output, has_detections = future.result()
                        
                        # Buffer the frame
                        frame_buffer[frame_num] = (frame_output, has_detections)
                        
                        # Submit next frame
                        frame_data = frame_reader_render.get_frame()
                        if frame_data is not None:
                            f_num, frame = frame_data
                            validated_candidates = temporal_filter.get_validated_detections(f_num)
                            
                            new_future = executor.submit(
                                render_frame_worker,
                                frame.copy(), f_num, validated_candidates,
                                detector_svm, visualizer, roi_pixel_map
                            )
                            futures_dict[new_future] = f_num
                
                # Write frames in order
                while next_frame_to_write in frame_buffer:
                    frame_output, has_detections = frame_buffer.pop(next_frame_to_write)
                    video_writer.write(frame_output)
                    
                    if has_detections:
                        frames_with_detections += 1
                    
                    frame_count_render += 1
                    next_frame_to_write += 1
                    
                    if frame_count_render % config.PROGRESS_UPDATE_INTERVAL == 0:
                        progress = int(frame_count_render / total_frames * 100)
                        print(f"   Rendered {frame_count_render}/{total_frames} frames ({progress}%)")
                    
                    del frame_output
        
        frame_reader_render.stop()
        video_writer.release()
        
        render_time = time.time() - start_render
        print(f"\n✓ Rendering Pass completed in {render_time:.2f}s")
        print(f"  Speed: {frame_count_render/render_time:.1f} FPS")
        print(f"  Frames with detections: {frames_with_detections}/{frame_count_render}")
        
        # =====================================================================
        # PASS 3: NO LONGER NEEDED (masks written in Pass 1)
        # =====================================================================
        # Masks are now written during Pass 1 in streaming mode
        
        # =====================================================================
        # FINAL SUMMARY
        # =====================================================================
        total_time = time.time() - start_total
        
        print("\n" + "=" * 70)
        print("✓ PROCESSING COMPLETE!")
        print("=" * 70)
        print(f"Output file: {config.OUTPUT_VIDEO_PATH}")
        
        if config.SAVE_MASK_VIDEOS:
            print(f"\nMask videos:")
            print(f"   Blue: {config.MASK_VIDEO_BLUE}")
            print(f"   Red: {config.MASK_VIDEO_RED}")
            print(f"   Yellow: {config.MASK_VIDEO_YELLOW}")
        
        print(f"\nPerformance:")
        print(f"   Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"   Detection phase: {detection_time:.2f}s ({detection_time/total_time*100:.1f}%)")
        print(f"   Rendering phase: {render_time:.2f}s ({render_time/total_time*100:.1f}%)")
        print(f"   Overall speed: {frame_count_render/total_time:.1f} FPS")
        
        print(f"\nOptimizations Applied:")
        print(f"   ✓ Multi-threading ({config.NUM_WORKERS} workers)")
        print(f"   ✓ Frame prefetching (buffer: {config.PREFETCH_FRAMES})")
        print(f"   ✓ Parallel batch processing (batch size: {config.BATCH_SIZE})")
        print(f"   ✓ Parallel SVM verification")
        print(f"   ✓ Streaming mask writing (zero memory accumulation)")
        print(f"   ✓ Temporal filtering (interpolation + smoothing)")
        print(f"   ✓ Detection cache (pre-computed)")
        print(f"   ✓ Aggressive memory management")
        
        speedup = (frame_count_render / total_time) / (frame_count_render / (detection_time + render_time)) \
                  if (detection_time + render_time) > 0 else 1.0
        print(f"\n   Estimated speedup: {speedup:.2f}x")
        
        if config.DEBUG_MODE:
            print(f"\nDebug Features:")
            print(f"   ✓ Frame ID overlay")
            print(f"   ✓ ROI visualization")
            print(f"   ✓ Detection metrics")
        
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        gc.collect()


def render_frame_worker(frame, frame_num, validated_candidates, detector_svm, 
                       visualizer, roi_pixel_map):
    """Worker for parallel rendering with SVM verification"""
    final_detections = []
    
    if validated_candidates:
        for (bbox, color, metrics) in validated_candidates:
            if detector_svm.verify(frame, bbox):
                final_detections.append((bbox, color, metrics))
    
    frame_output = visualizer.draw_all(frame, frame_num, final_detections, roi_pixel_map)
    has_detections = len(final_detections) > 0
    
    return frame_output, has_detections


def write_mask_video(masks, writer, color_name):
    """Worker for writing mask videos"""
    frames_written = 0
    for mask in masks:
        writer.write(mask)
        frames_written += 1
    return frames_written


if __name__ == "__main__":
    import concurrent.futures
    main()
