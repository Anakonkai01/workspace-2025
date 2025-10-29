import cv2 
import numpy as np 
import time
from collections import defaultdict
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
from threading import Lock, Thread
from multiprocessing import cpu_count, Manager, Queue, Process
import os

# =============================================================================
# GLOBAL VARIABLES (Hard-coded Parameters)
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

# --- Temporal Filtering Parameters ---
MIN_DURATION_SEC = 2.0
MAX_GAP_SEC = 0.5
IOU_THRESHOLD = 0.3

# --- Performance Parameters (AUTO-DETECTED) ---
CPU_CORES = cpu_count()
NUM_READ_THREADS = min(4, CPU_CORES // 8)           # 4 reader threads
NUM_PROCESS_WORKERS = CPU_CORES - NUM_READ_THREADS - 2  # Use remaining cores for processing
FRAME_BUFFER_SIZE = 120                             # Larger buffer for 32 cores
BATCH_SIZE = 30                                     # Process frames in batches

print(f"🖥️  Detected {CPU_CORES} CPU cores")
print(f"⚡ Using {NUM_READ_THREADS} readers + {NUM_PROCESS_WORKERS} processors")


# =============================================================================
# TEMPORAL FILTER CLASS (OPTIMIZED)
# =============================================================================

class TemporalSignFilter:
    """Filter traffic signs based on temporal consistency"""
    
    def __init__(self, fps, min_duration_sec=3.0, max_gap_sec=0.5, iou_threshold=0.3):
        self.fps = fps
        self.min_frames = int(min_duration_sec * fps)
        self.max_gap_frames = int(max_gap_sec * fps)
        self.iou_threshold = iou_threshold
        
        self.tracks = defaultdict(list)
        self.next_track_id = 0
        
        # Cache for validated detections (OPTIMIZATION)
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
        """Add detections from current frame"""
        for bbox, color in detections:
            best_match_id = None
            best_iou = 0
            
            for track_id, track_data in self.tracks.items():
                if len(track_data) == 0:
                    continue
                    
                last_detection = track_data[-1]
                
                if (last_detection['color'] == color and 
                    frame_num - last_detection['frame'] <= self.max_gap_frames):
                    
                    iou = self.calculate_iou(bbox, last_detection['bbox'])
                    if iou > best_iou and iou >= self.iou_threshold:
                        best_iou = iou
                        best_match_id = track_id
            
            if best_match_id is not None:
                self.tracks[best_match_id].append({
                    'frame': frame_num,
                    'bbox': bbox,
                    'color': color
                })
            else:
                self.tracks[self.next_track_id] = [{
                    'frame': frame_num,
                    'bbox': bbox,
                    'color': color
                }]
                self.next_track_id += 1
    
    def interpolate_missing_frames(self, track_data):
        """Fill missing frames with linear interpolation"""
        if len(track_data) < 2:
            return track_data
        
        interpolated = []
        
        for i in range(len(track_data) - 1):
            current = track_data[i]
            next_det = track_data[i + 1]
            
            interpolated.append(current)
            
            frame_gap = next_det['frame'] - current['frame']
            
            if 1 < frame_gap <= self.max_gap_frames:
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
        🚀 OPTIMIZATION: Pre-build cache of all validated detections
        This eliminates repeated computation in Pass 2
        """
        print("   🔧 Building detection cache...")
        start = time.time()
        
        for track_id, track_data in self.tracks.items():
            if len(track_data) == 0:
                continue
            
            first_frame = track_data[0]['frame']
            last_frame = track_data[-1]['frame']
            duration_frames = last_frame - first_frame + 1
            
            if duration_frames >= self.min_frames:
                interpolated = self.interpolate_missing_frames(track_data)
                smoothed = self.smooth_bounding_boxes(interpolated, window_size=5)
                
                for detection in smoothed:
                    frame_num = detection['frame']
                    if frame_num not in self._validated_cache:
                        self._validated_cache[frame_num] = []
                    self._validated_cache[frame_num].append(
                        (detection['bbox'], detection['color'])
                    )
        
        self._cache_built = True
        print(f"   ✅ Cache built in {time.time() - start:.2f}s ({len(self._validated_cache)} frames)")
    
    def get_validated_detections(self, frame_num):
        """Get validated detections (using cache)"""
        if not self._cache_built:
            self.build_detection_cache()
        
        return self._validated_cache.get(frame_num, [])
    
    def get_statistics(self):
        """Get tracking statistics"""
        total_tracks = len(self.tracks)
        valid_tracks = sum(1 for track in self.tracks.values() 
                           if len(track) > 0 and 
                           (track[-1]['frame'] - track[0]['frame'] + 1) >= self.min_frames)
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
    """Preprocess frame"""
    blur_ksize = max(3, blur_ksize)
    if blur_ksize % 2 == 0:
        blur_ksize += 1
        
    frame_processing = cv2.medianBlur(frame, blur_ksize) 
    frame_processing = cv2.cvtColor(frame_processing, cv2.COLOR_BGR2HSV)      
    
    h, s, v = cv2.split(frame_processing)  
    clahe = cv2.createCLAHE(clipLimit=clip_limit / 10.0, tileGridSize=(1,1))
    v_clahe = clahe.apply(v)
    s = s.astype(np.float32) * 1.5
    s = np.clip(s, 0, 255).astype(np.uint8)
    hsv_blur_clahe = cv2.merge([h, s, v_clahe])
    return hsv_blur_clahe

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
# DETECTION FUNCTIONS
# =============================================================================

def extract_circle_detections(mask, roi_params, color_type):
    """Extract circular detections"""
    detections = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour) 
        
        if area < 250 or area > 15000:
            continue
        
        hull = cv2.convexHull(contour)
        perimeter_hull = cv2.arcLength(hull, True) 
        area_hull = cv2.contourArea(hull)
        x, y, w, h = cv2.boundingRect(hull)
        
        is_in_roi = is_bbox_in_roi((x, y, w, h), roi_params, overlap_threshold=0.5)

        if perimeter_hull > 0:
            circularity = 4 * np.pi * (area_hull / (perimeter_hull * perimeter_hull))
            
            circularity_param = 0.87
            if area < 725 and not is_in_roi:
                continue
            elif area < 725 and is_in_roi:
                circularity_param = 0.87
            elif area >= 725:
                circularity_param = 0.93

            if circularity > circularity_param:
                detections.append(((x, y, w, h), color_type))
                
    return detections

def extract_triangle_detections(mask, roi_params, color_type):
    """Extract triangular detections"""
    detections = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < 825: 
            continue
            
        x, y, w, h = cv2.boundingRect(contour)

        is_in_roi = is_bbox_in_roi((x, y, w, h), roi_params, overlap_threshold=0.5)
        
        if not is_in_roi:
            continue
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        if hull_area == 0:
            continue
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        solidity = float(area) / hull_area
        
        if solidity <= 0.75: 
            continue

        epsilon = 0.03 * perimeter 
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)
        
        if num_vertices <= 7:
            detections.append(((x, y, w, h), color_type))
            
    return detections

def draw_detections(frame, detections):
    """Draw bounding boxes"""
    color_map = {
        'blue': (255, 0, 0),
        'red': (0, 0, 255),
        'yellow': (0, 255, 255)
    }
    
    for bbox, color_type in detections:
        x, y, w, h = bbox
        color_bgr = color_map.get(color_type, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 3)
    
    return frame


# =============================================================================
# 🚀 OPTIMIZED PASS 2 - THREADED VIDEO PROCESSING
# =============================================================================

def frame_reader_worker(video_path, frame_queue, start_frame, end_frame):
    """
    🚀 Thread worker for reading frames
    Reads frames in batches and puts them in queue
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_queue.put((frame_num, frame.copy()))
    
    cap.release()

def frame_processor_worker(input_queue, output_queue, temporal_filter):
    """
    🚀 Thread worker for processing frames
    Draws detections on frames
    """
    while True:
        try:
            item = input_queue.get(timeout=1)
            if item is None:  # Poison pill
                break
            
            frame_num, frame = item
            
            # Get validated detections (from cache)
            validated = temporal_filter.get_validated_detections(frame_num)
            
            # Draw detections
            frame_output = draw_detections(frame, validated)
            
            # Put to output queue
            output_queue.put((frame_num, frame_output))
            
        except queue.Empty:
            continue


# =============================================================================
# 🚀 MULTIPROCESSING FUNCTIONS FOR PASS 1
# =============================================================================

def process_frame_batch(args):
    """
    Process a batch of frames in parallel (Pass 1)
    This runs in separate processes to bypass GIL
    """
    frames_data, height_new, width_new = args
    
    # ROI parameters (recreate in each process)
    blue_roi_params = (int(width_new * 0.45), 0, width_new, height_new)
    red_roi_params = (int(width_new * 0.45), 0, width_new, int(height_new * 0.94))
    yellow_roi_params = (int(width_new * 0.45), int(height_new * 0.5), width_new, int(height_new * 0.9))
    
    batch_results = []
    
    for frame_num, frame_full in frames_data:
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
        
        # Extract detections
        all_detections = []
        all_detections.extend(extract_circle_detections(mask_blue_clean, blue_roi_params, 'blue'))
        all_detections.extend(extract_circle_detections(mask_red_clean, red_roi_params, 'red'))
        all_detections.extend(extract_triangle_detections(mask_yellow_clean, yellow_roi_params, 'yellow'))
        
        batch_results.append((frame_num, all_detections))
    
    return batch_results

def optimized_pass2(input_video, output_video, temporal_filter, total_frames, fps, resolution):
    """
    🚀 OPTIMIZED PASS 2 with multi-threading
    
    Architecture:
    1. Reader threads: Read frames from video
    2. Processor threads: Draw detections (using cache)
    3. Writer thread (main): Write frames to output video in order
    
    Speed improvement: ~3-4x faster than sequential
    """
    print(f"\n🎨 PASS 2: Ultra-fast video rendering (multi-threaded)...")
    print(f"   🔧 Threads: {NUM_READ_THREADS} readers + {NUM_PROCESS_WORKERS} processors")
    start_pass2 = time.time()
    
    # Build detection cache first (if not already built)
    if not temporal_filter._cache_built:
        temporal_filter.build_detection_cache()
    
    # Create queues
    read_queue = queue.Queue(maxsize=FRAME_BUFFER_SIZE)
    process_queue = queue.Queue(maxsize=FRAME_BUFFER_SIZE)
    
    # Start reader thread
    frames_per_reader = total_frames // NUM_READ_THREADS
    reader_threads = []
    
    for i in range(NUM_READ_THREADS):
        start_frame = i * frames_per_reader
        end_frame = (i + 1) * frames_per_reader if i < NUM_READ_THREADS - 1 else total_frames
        
        from threading import Thread
        t = Thread(target=frame_reader_worker, 
                   args=(input_video, read_queue, start_frame, end_frame))
        t.start()
        reader_threads.append(t)
    
    # Start processor threads
    processor_threads = []
    for _ in range(NUM_PROCESS_WORKERS):
        from threading import Thread
        t = Thread(target=frame_processor_worker, 
                   args=(read_queue, process_queue, temporal_filter))
        t.start()
        processor_threads.append(t)
    
    # Writer (main thread) - writes frames in order
    w, h = resolution
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
    
    if not video_writer.isOpened():
        print(f"❌ Error: Cannot create output file '{output_video}'")
        return False
    
    # Buffer for out-of-order frames
    frame_buffer = {}
    next_frame_to_write = 0
    frames_written = 0
    
    # Progress tracking
    last_progress = 0
    
    while frames_written < total_frames:
        try:
            frame_num, frame_output = process_queue.get(timeout=5)
            frame_buffer[frame_num] = frame_output
            
            # Write frames in order
            while next_frame_to_write in frame_buffer:
                video_writer.write(frame_buffer[next_frame_to_write])
                del frame_buffer[next_frame_to_write]
                next_frame_to_write += 1
                frames_written += 1
                
                # Progress
                progress = int(frames_written / total_frames * 100)
                if progress >= last_progress + 10:
                    print(f"   🎬 Rendered {frames_written}/{total_frames} frames ({progress}%)")
                    last_progress = progress
                    
        except queue.Empty:
            # Check if all readers finished
            if all(not t.is_alive() for t in reader_threads):
                break
    
    # Wait for all threads to finish
    for t in reader_threads:
        t.join()
    
    # Send poison pills to processors
    for _ in range(NUM_PROCESS_WORKERS):
        read_queue.put(None)
    
    for t in processor_threads:
        t.join()
    
    video_writer.release()
    end_pass2 = time.time()
    
    print(f"   ✅ Pass 2 completed in {end_pass2 - start_pass2:.2f}s")
    print(f"   ⚡ Rendering speed: {total_frames/(end_pass2 - start_pass2):.1f} FPS")
    
    return True


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main():
    input_video = 'task1.mp4'
    final_output = 'detected_signs_ultra_fast.mp4'
    
    # Check video
    cap_test = cv2.VideoCapture(input_video)
    if not cap_test.isOpened():
        print(f"❌ Error: Cannot open video '{input_video}'")
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
    print("   🚀  ULTRA-FAST TRAFFIC SIGN DETECTION")
    print("=" * 70)
    print(f"📹 Video: {input_video}")
    print(f"📐 Resolution: {w_orig}x{h_orig} @ {fps:.2f} FPS")
    print(f"🎞️  Total frames: {total_frames}")
    print(f"⏱️  Minimum duration: {MIN_DURATION_SEC}s ({int(MIN_DURATION_SEC * fps)} frames)")
    print(f"⚡ Flicker tolerance: {MAX_GAP_SEC}s ({int(MAX_GAP_SEC * fps)} frames)")
    print(f"\n🔧 OPTIMIZATION:")
    print(f"   • Detection cache: Pre-computed")
    print(f"   • Multi-processing: {NUM_PROCESS_WORKERS} worker processes (Pass 1)")
    print(f"   • Multi-threading: {NUM_READ_THREADS + NUM_PROCESS_WORKERS} threads (Pass 2)")
    print(f"   • Frame buffer: {FRAME_BUFFER_SIZE} frames")
    print(f"   • Batch size: {BATCH_SIZE} frames/batch")
    print("=" * 70)
    
    # =========================================================================
    # PASS 1: DETECTION (MULTIPROCESSING)
    # =========================================================================
    
    print("\n🔍 PASS 1: Detecting traffic signs (multi-processing)...")
    start_pass1 = time.time()
    
    temporal_filter = TemporalSignFilter(
        fps, 
        min_duration_sec=MIN_DURATION_SEC,
        max_gap_sec=MAX_GAP_SEC,
        iou_threshold=IOU_THRESHOLD
    )
    
    # Read ALL frames first (memory permitting) or in large batches
    print("   📖 Reading video frames...")
    cap = cv2.VideoCapture(input_video)
    
    height_new = int(h_orig * 0.475)
    width_new = w_orig
    
    all_frames = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame_full = cap.read()
        if not ret:
            break
        all_frames.append((frame_count, frame_full))
        frame_count += 1
    
    cap.release()
    print(f"   ✅ Loaded {len(all_frames)} frames into memory")
    
    # Split frames into batches for parallel processing
    print(f"   🔧 Processing with {NUM_PROCESS_WORKERS} parallel workers...")
    
    batches = []
    for i in range(0, len(all_frames), BATCH_SIZE):
        batch = all_frames[i:i+BATCH_SIZE]
        batches.append((batch, height_new, width_new))
    
    # Process batches in parallel using multiprocessing
    all_detections_results = []
    
    with ProcessPoolExecutor(max_workers=NUM_PROCESS_WORKERS) as executor:
        futures = {executor.submit(process_frame_batch, batch): i for i, batch in enumerate(batches)}
        
        completed = 0
        for future in as_completed(futures):
            batch_results = future.result()
            all_detections_results.extend(batch_results)
            completed += 1
            
            if completed % 10 == 0 or completed == len(batches):
                progress = int(completed / len(batches) * 100)
                print(f"   📊 Processed {completed}/{len(batches)} batches ({progress}%)")
    
    # Sort results by frame number and add to temporal filter
    all_detections_results.sort(key=lambda x: x[0])
    
    print("   🔗 Building temporal tracks...")
    for frame_num, detections in all_detections_results:
        temporal_filter.add_detections(frame_num, detections)
    
    end_pass1 = time.time()
    
    # Statistics
    total_tracks, valid_tracks = temporal_filter.get_statistics()
    print(f"\n✅ PASS 1 completed in {end_pass1 - start_pass1:.2f}s")
    print(f"📊 Statistics:")
    print(f"   • Total tracks: {total_tracks}")
    print(f"   • Valid tracks: {valid_tracks}")
    print(f"   • Filtered: {total_tracks - valid_tracks}")
    
    # =========================================================================
    # PASS 2: OPTIMIZED RENDERING
    # =========================================================================
    
    success = optimized_pass2(
        input_video, 
        final_output, 
        temporal_filter, 
        total_frames, 
        fps, 
        (w_orig, h_orig)
    )
    
    if not success:
        return
    
    # =========================================================================
    # COMPLETION
    # =========================================================================
    
    total_time = time.time() - start_pass1
    
    print("\n" + "=" * 70)
    print("✅  PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"📁 Output: {final_output}")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"⚡ Overall speed: {total_frames/total_time:.1f} FPS")
    
    if total_tracks > 0:
        print(f"\n📈 Filtering:")
        print(f"   • Retention: {valid_tracks/total_tracks*100:.1f}%")
        
    print("\n🚀 Optimizations applied:")
    print(f"   ✓ Detection cache (pre-computed)")
    print(f"   ✓ Multi-processing Pass 1 ({NUM_PROCESS_WORKERS} processes)")
    print(f"   ✓ Multi-threaded Pass 2 ({NUM_READ_THREADS + NUM_PROCESS_WORKERS} threads)")
    print(f"   ✓ Interpolation + Smoothing")
    print(f"   ✓ Frame buffering ({FRAME_BUFFER_SIZE} frames)")
    print(f"   ✓ Batch processing ({BATCH_SIZE} frames/batch)")
    print("=" * 70)


if __name__ == "__main__":
    main()