import cv2 
import numpy as np 
import time
from collections import defaultdict
import pickle
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
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

# --- Performance Parameters ---
NUM_WORKERS = max(1, cpu_count() - 1)  # Số worker = số core - 1
BATCH_SIZE = 30  # Số frame xử lý cùng lúc


# =============================================================================
# TEMPORAL FILTER CLASS
# =============================================================================

class TemporalSignFilter:
    """Lọc biển báo dựa trên tính nhất quán theo thời gian"""
    
    def __init__(self, fps, min_duration_sec=3.0, max_gap_sec=0.5, iou_threshold=0.3):
        self.fps = fps
        self.min_frames = int(min_duration_sec * fps)
        self.max_gap_frames = int(max_gap_sec * fps)
        self.iou_threshold = iou_threshold
        
        self.tracks = defaultdict(list)
        self.next_track_id = 0
        
    def calculate_iou(self, box1, box2):
        """Tính IoU giữa 2 bounding box"""
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
        """Thêm detections từ frame hiện tại"""
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
        """Lấp đầy các khoảng trống bằng linear interpolation"""
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
        """Làm mượt bbox bằng moving average"""
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
    
    def get_validated_detections(self, frame_num):
        """Lấy detections đã validate cho frame cụ thể"""
        validated = []
        
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
                    if detection['frame'] == frame_num:
                        validated.append((detection['bbox'], detection['color']))
        
        return validated
    
    def get_statistics(self):
        """Thống kê tracking"""
        total_tracks = len(self.tracks)
        valid_tracks = sum(1 for track in self.tracks.values() 
                           if len(track) > 0 and 
                           (track[-1]['frame'] - track[0]['frame'] + 1) >= self.min_frames)
        return total_tracks, valid_tracks


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def morphology(mask, k_size, iter_opening, iter_close):
    """Phép toán hình thái học"""
    k_size = max(1, k_size)
    if k_size % 2 == 0:
        k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(k_size, k_size)) 
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iter_opening)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=iter_close)
    return mask_clean

def preprocess_frame(frame, clip_limit, blur_ksize):
    """Tiền xử lý frame"""
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
    """
    Kiểm tra bbox có trong ROI không
    Tính tỷ lệ overlap, nếu >= threshold → True
    """
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
    """Trích xuất biển báo hình tròn"""
    detections = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour) 
        
        if area < 100 or area > 15000:
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
    """Trích xuất biển báo tam giác"""
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


# =============================================================================
# PARALLEL PROCESSING FUNCTIONS - CORE OPTIMIZATION
# =============================================================================

def process_single_frame(args):
    """
    ⚡ XỬ LÝ 1 FRAME - ĐƯỢC CHẠY SONG SONG
    
    Hàm này được gọi bởi nhiều worker cùng lúc
    Mỗi worker xử lý 1 frame độc lập
    
    Args:
        args: tuple (frame_data, frame_num, crop_params, roi_params)
    
    Returns:
        tuple: (frame_num, all_detections)
    """
    frame_data, frame_num, crop_params, all_roi_params = args
    
    # Giải nén frame từ bytes (để truyền qua process)
    frame_full = pickle.loads(frame_data)
    
    # Crop vùng xử lý
    h_orig, width_new, height_new = crop_params
    frame_to_process = frame_full[0:height_new, 0:width_new]
    
    blue_roi, red_roi, yellow_roi = all_roi_params
    
    # ===== XỬ LÝ BLUE =====
    hsv_blue = preprocess_frame(frame_to_process, blue_clahe_clip_limit, blue_blur_ksize)
    lower_blue = np.array([blue_lower_h, blue_lower_s, blue_lower_v])
    upper_blue = np.array([blue_upper_h, blue_upper_s, blue_upper_v])
    mask_blue = cv2.inRange(hsv_blue, lower_blue, upper_blue)
    mask_blue_clean = morphology(mask_blue, blue_ksize, blue_open_iter, blue_close_iter)
    
    # ===== XỬ LÝ RED =====
    hsv_red = preprocess_frame(frame_to_process, red_clahe_clip_limit, red_blur_ksize)
    lower_red = np.array([red_lower_h, red_lower_s, red_lower_v])
    upper_red = np.array([red_upper_h, red_upper_s, red_upper_v])
    mask_red = cv2.inRange(hsv_red, lower_red, upper_red)
    mask_red_clean = morphology(mask_red, red_ksize, red_open_iter, red_close_iter)
    
    # ===== XỬ LÝ YELLOW =====
    hsv_yellow = preprocess_frame(frame_to_process, yellow_clahe_clip_limit, yellow_blur_ksize)
    lower_yellow = np.array([yellow_lower_h, yellow_lower_s, yellow_lower_v])
    upper_yellow = np.array([yellow_upper_h, yellow_upper_s, yellow_upper_v])
    mask_yellow = cv2.inRange(hsv_yellow, lower_yellow, upper_yellow)
    mask_yellow_clean = morphology(mask_yellow, yellow_ksize, yellow_open_iter, yellow_close_iter)
    
    # ===== TRÍCH XUẤT DETECTIONS =====
    all_detections = []
    all_detections.extend(extract_circle_detections(mask_blue_clean, blue_roi, 'blue'))
    all_detections.extend(extract_circle_detections(mask_red_clean, red_roi, 'red'))
    all_detections.extend(extract_triangle_detections(mask_yellow_clean, yellow_roi, 'yellow'))
    
    return (frame_num, all_detections)


def read_frames_batch(cap, start_frame, batch_size, crop_params):
    """
    ⚡ ĐỌC BATCH FRAMES - GIẢM I/O OVERHEAD
    
    Đọc nhiều frame cùng lúc thay vì đọc từng frame
    → Giảm số lần gọi cap.read()
    
    Args:
        cap: VideoCapture object
        start_frame: Frame bắt đầu
        batch_size: Số frame cần đọc
        crop_params: (h_orig, width_new, height_new)
    
    Returns:
        list: [(frame_data, frame_num), ...]
    """
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for i in range(batch_size):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Serialize frame để truyền qua process
        frame_data = pickle.dumps(frame)
        frames.append((frame_data, start_frame + i))
    
    return frames


def draw_detections(frame, detections):
    """Vẽ bounding box"""
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
# MAIN PROGRAM - PARALLEL TWO PASS PROCESSING
# =============================================================================

def main():
    input_video = 'task1.mp4'
    final_output = 'detected_signs_filtered_100a.mp4'
    
    # Kiểm tra video
    cap_test = cv2.VideoCapture(input_video)
    if not cap_test.isOpened():
        print(f"❌ Error: Cannot open video '{input_video}'")
        return
    cap_test.release()
    
    # Lấy thông số video
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: 
        fps = 30.0
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Tính toán parameters
    height_new = int(h_orig * 0.475)
    width_new = w_orig
    crop_params = (h_orig, width_new, height_new)
    
    # ROI cho từng màu
    blue_roi_params = (int(width_new * 0.45), 0, width_new, height_new)
    red_roi_params = (int(width_new * 0.45), 0, width_new, int(height_new * 0.94))
    yellow_roi_params = (int(width_new * 0.45), int(height_new * 0.5), width_new, int(height_new * 0.9))
    all_roi_params = (blue_roi_params, red_roi_params, yellow_roi_params)
    
    print("=" * 70)
    print("   🚀  TRAFFIC SIGN DETECTION - PARALLEL PROCESSING")
    print("=" * 70)
    print(f"📹 Video: {input_video}")
    print(f"📐 Resolution: {w_orig}x{h_orig} @ {fps:.2f} FPS")
    print(f"🎞️  Total frames: {total_frames}")
    print(f"⏱️  Minimum duration: {MIN_DURATION_SEC}s ({int(MIN_DURATION_SEC * fps)} frames)")
    print(f"⚡ Flicker tolerance: {MAX_GAP_SEC}s ({int(MAX_GAP_SEC * fps)} frames)")
    print(f"\n🔧 PERFORMANCE SETTINGS:")
    print(f"   • CPU cores available: {cpu_count()}")
    print(f"   • Worker processes: {NUM_WORKERS}")
    print(f"   • Batch size: {BATCH_SIZE} frames")
    print(f"   • Expected speedup: ~{NUM_WORKERS}x")
    print("=" * 70)
    
    # =========================================================================
    # PASS 1: PARALLEL DETECTION
    # =========================================================================
    
    print("\n🔍 PASS 1: Parallel detection with multiprocessing...")
    start_pass1 = time.time()
    
    temporal_filter = TemporalSignFilter(
        fps, 
        min_duration_sec=MIN_DURATION_SEC,
        max_gap_sec=MAX_GAP_SEC,
        iou_threshold=IOU_THRESHOLD
    )
    
    cap = cv2.VideoCapture(input_video)
    
    # Xử lý theo batch
    num_batches = (total_frames + BATCH_SIZE - 1) // BATCH_SIZE
    
    with Pool(processes=NUM_WORKERS) as pool:
        for batch_idx in range(num_batches):
            start_frame = batch_idx * BATCH_SIZE
            current_batch_size = min(BATCH_SIZE, total_frames - start_frame)
            
            # Đọc batch frames
            frames_batch = read_frames_batch(cap, start_frame, current_batch_size, crop_params)
            
            if not frames_batch:
                break
            
            # Chuẩn bị args cho parallel processing
            args_list = [
                (frame_data, frame_num, crop_params, all_roi_params)
                for frame_data, frame_num in frames_batch
            ]
            
            # ⚡ XỬ LÝ SONG SONG - CORE MAGIC HERE
            results = pool.map(process_single_frame, args_list)
            
            # Thu thập kết quả
            for frame_num, detections in results:
                temporal_filter.add_detections(frame_num, detections)
            
            # Progress
            processed = min(start_frame + BATCH_SIZE, total_frames)
            print(f"   📊 Processed {processed}/{total_frames} frames " 
                  f"({processed/total_frames*100:.1f}%)")
    
    cap.release()
    end_pass1 = time.time()
    
    # Thống kê
    total_tracks, valid_tracks = temporal_filter.get_statistics()
    print(f"\n✅ PASS 1 completed in {end_pass1 - start_pass1:.2f}s")
    print(f"   ⚡ Processing speed: {total_frames/(end_pass1 - start_pass1):.1f} FPS")
    print(f"📊 Statistics:")
    print(f"   • Total tracks: {total_tracks}")
    print(f"   • Valid tracks: {valid_tracks}")
    print(f"   • Filtered: {total_tracks - valid_tracks}")
    
    # =========================================================================
    # PASS 2: RENDER VIDEO (Sequential - vì VideoWriter không thread-safe)
    # =========================================================================
    
    print(f"\n🎨 PASS 2: Rendering video...")
    start_pass2 = time.time()
    
    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(final_output, fourcc, fps, (w_orig, h_orig))
    
    if not video_writer.isOpened():
        print(f"❌ Error: Cannot create output file '{final_output}'")
        cap.release()
        return
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame_full = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"   🎬 Rendered {frame_count}/{total_frames} frames...")
        
        validated = temporal_filter.get_validated_detections(frame_count - 1)
        frame_output = draw_detections(frame_full.copy(), validated)
        video_writer.write(frame_output)
    
    cap.release()
    video_writer.release()
    end_pass2 = time.time()
    
    # =========================================================================
    # COMPLETION
    # =========================================================================
    
    total_time = end_pass2 - start_pass1
    
    print("\n" + "=" * 70)
    print("✅  PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"📁 Output: {final_output}")
    print(f"⏱️  Pass 1 (Parallel): {end_pass1 - start_pass1:.2f}s")
    print(f"⏱️  Pass 2 (Sequential): {end_pass2 - start_pass2:.2f}s")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"⚡ Average FPS: {total_frames/total_time:.1f}")
    
    if total_tracks > 0:
        print(f"\n📈 Filtering efficiency:")
        print(f"   • Retention: {valid_tracks/total_tracks*100:.1f}%")
        
    print("\n🔧 Techniques applied:")
    print(f"   ✓ Multiprocessing ({NUM_WORKERS} workers)")
    print(f"   ✓ Batch processing ({BATCH_SIZE} frames/batch)")
    print(f"   ✓ Interpolation + Smoothing")
    print("=" * 70)


if __name__ == "__main__":
    main()