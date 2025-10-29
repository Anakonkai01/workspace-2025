import cv2 
import numpy as np 
import time
from collections import defaultdict
import pickle

# =============================================================================
# === BIẾN TOÀN CỤC (Hard-coded) ===
# =============================================================================

# --- Thông số cho MÀU XANH DƯƠNG (Blue) ---
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

# --- Thông số cho MÀU ĐỎ (Red) ---
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

# --- Thông số cho MÀU VÀNG (Yellow) ---
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

# --- Thông số Temporal Filtering ---
MIN_DURATION_SEC = 2.0      # Thời gian tối thiểu để coi là biển báo thật
MAX_GAP_SEC = 0.5       # Khoảng trống tối đa cho phép (xử lý flicker)
IOU_THRESHOLD = 0.3         # Ngưỡng IoU để match objects


# =============================================================================
# === CLASS TEMPORAL FILTER ===
# =============================================================================

class TemporalSignFilter:
    """Lọc biển báo dựa trên temporal consistency"""
    
    def __init__(self, fps, min_duration_sec=3.0, max_gap_sec=0.5, iou_threshold=0.3):
        self.fps = fps
        self.min_frames = int(min_duration_sec * fps)
        self.max_gap_frames = int(max_gap_sec * fps)
        self.iou_threshold = iou_threshold
        
        self.tracks = defaultdict(list)
        self.next_track_id = 0
        
    def calculate_iou(self, box1, box2):
        """Tính IoU giữa 2 bounding boxes (x, y, w, h)"""
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
    
    def get_validated_detections(self, frame_num):
        """Lấy các detections đã được validate cho frame cụ thể"""
        validated = []
        
        for track_id, track_data in self.tracks.items():
            if len(track_data) == 0:
                continue
            
            first_frame = track_data[0]['frame']
            last_frame = track_data[-1]['frame']
            duration_frames = last_frame - first_frame + 1
            
            if duration_frames >= self.min_frames:
                for detection in track_data:
                    if detection['frame'] == frame_num:
                        validated.append((detection['bbox'], detection['color']))
        
        return validated
    
    def get_statistics(self):
        """Lấy thống kê về tracking"""
        total_tracks = len(self.tracks)
        valid_tracks = sum(1 for track in self.tracks.values() 
                           if len(track) > 0 and 
                           (track[-1]['frame'] - track[0]['frame'] + 1) >= self.min_frames)
        return total_tracks, valid_tracks


# =============================================================================
# === HÀM HỖ TRỢ CHUNG ===
# =============================================================================

def morphology(mask, k_size, iter_opening, iter_close):
    k_size = max(1, k_size)
    if k_size % 2 == 0:
        k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(k_size, k_size)) 
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iter_opening)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=iter_close)
    return mask_clean

def preprocess_frame(frame, clip_limit, blur_ksize):
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


# =============================================================================
# === HÀM PHÁT HIỆN VÀ TRÍCH XUẤT DETECTIONS ===
# =============================================================================

def extract_circle_detections(mask, roi_params, color_type):
    """Trích xuất detections hình tròn từ mask"""
    (roi_x_start, roi_y_start, roi_x_end, roi_y_end) = roi_params
    detections = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour) 
        if area < 800 or area > 15000:
            continue
        
        hull = cv2.convexHull(contour)
        perimeter_hull = cv2.arcLength(hull, True) 
        area_hull = cv2.contourArea(hull)
        x, y, w, h = cv2.boundingRect(hull)
        
        cx = x + w // 2
        cy = y + h // 2
        is_in_roi = (cx >= roi_x_start and cx <= roi_x_end and
                     cy >= roi_y_start and cy <= roi_y_end)

        if perimeter_hull > 0:
            circularity = 4 * np.pi * (area_hull / (perimeter_hull * perimeter_hull))
            circularity_param = 0.87
            if area < 725 and is_in_roi is False:
                continue 
            if area < 725 and is_in_roi:
                circularity_param = 0.87
            elif area >= 725:
                circularity_param = 0.93

            if circularity > circularity_param:
                detections.append(((x, y, w, h), color_type))
                
    return detections

def extract_triangle_detections(mask, roi_params, color_type):
    """(CHỈNH SỬA) Trích xuất detections hình tam giác từ mask (ĐÃ THÊM ROI)"""
    (roi_x_start, roi_y_start, roi_x_end, roi_y_end) = roi_params
    detections = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Ngưỡng diện tích (điều chỉnh nếu cần)
        if area < 825: 
            continue
            
        x, y, w, h = cv2.boundingRect(contour)

        cx = x + w // 2
        cy = y + h // 2
        is_in_roi = (cx >= roi_x_start and cx <= roi_x_end and
                     cy >= roi_y_start and cy <= roi_y_end)
        
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
        
        # Ngưỡng độ "đặc" (điều chỉnh nếu cần, vd: 0.84)
        if solidity <= 0.75: 
            continue

        epsilon = 0.03 * perimeter 
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)
        
        if num_vertices <= 7: # Lọc hình tam giác
            detections.append(((x, y, w, h), color_type))
            
    return detections

def draw_detections(frame, detections):
    """Vẽ bounding boxes lên frame"""
    color_map = {
        'blue': (255, 0, 0),    # BGR
        'red': (0, 0, 255),
        'yellow': (0, 255, 255) # BGR
    }
    
    for bbox, color_type in detections:
        x, y, w, h = bbox
        color_bgr = color_map.get(color_type, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 3)
    
    return frame


# =============================================================================
# === CHƯƠNG TRÌNH CHÍNH - TWO PASS PROCESSING ===
# =============================================================================

def main():
    input_video = 'task1.mp4'
    temp_output = 'temp_all_detections.pkl'  # File lưu trữ tạm
    final_output = 'detected_signs_filtered.mp4'
    
    # --- Kiểm tra file tồn tại ---
    cap_test = cv2.VideoCapture(input_video)
    if not cap_test.isOpened():
        print(f"❌ Lỗi: Không thể mở video '{input_video}'")
        return
    cap_test.release()
    
    # --- Lấy thông số video ---
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: 
        fps = 30.0
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print("=" * 60)
    print("        🚦  TRAFFIC SIGN DETECTION WITH TEMPORAL FILTERING")
    print("=" * 60)
    print(f"📹 Video: {input_video}")
    print(f"📐 Kích thước: {w_orig}x{h_orig} @ {fps:.2f} FPS")
    print(f"🎞️  Tổng frames: {total_frames}")
    print(f"⏱️  Thời gian tối thiểu: {MIN_DURATION_SEC}s ({int(MIN_DURATION_SEC * fps)} frames)")
    print(f"⚡ Flicker tolerance: {MAX_GAP_SEC}s ({int(MAX_GAP_SEC * fps)} frames)")
    print("=" * 60)
    
    # =========================================================================
    # === PASS 1: DETECT VÀ LƯU TẤT CẢ DETECTIONS ===
    # =========================================================================
    
    print("\n🔍 PASS 1: Phát hiện tất cả biển báo...")
    start_pass1 = time.time()
    
    temporal_filter = TemporalSignFilter(
        fps, 
        min_duration_sec=MIN_DURATION_SEC,
        max_gap_sec=MAX_GAP_SEC,
        iou_threshold=IOU_THRESHOLD
    )
    
    cap = cv2.VideoCapture(input_video)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame_full = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # In tiến độ
        if frame_count % 100 == 0:
            print(f"   📊 Đã xử lý {frame_count}/{total_frames} frames...")
        
        # Cắt vùng xử lý (47.5% phía trên)
        height_new = int(h_orig * 0.475)
        width_new = w_orig
        frame_to_process = frame_full[0:height_new, 0:width_new]
        
        # === XỬ LÝ MÀU XANH DƯƠNG ===
        hsv_blue = preprocess_frame(frame_to_process, blue_clahe_clip_limit, blue_blur_ksize)
        lower_blue = np.array([blue_lower_h, blue_lower_s, blue_lower_v])
        upper_blue = np.array([blue_upper_h, blue_upper_s, blue_upper_v])
        mask_blue = cv2.inRange(hsv_blue, lower_blue, upper_blue)
        mask_blue_clean = morphology(mask_blue, blue_ksize, blue_open_iter, blue_close_iter)
        
        # === XỬ LÝ MÀU ĐỎ ===
        hsv_red = preprocess_frame(frame_to_process, red_clahe_clip_limit, red_blur_ksize)
        lower_red = np.array([red_lower_h, red_lower_s, red_lower_v])
        upper_red = np.array([red_upper_h, red_upper_s, red_upper_v])
        mask_red = cv2.inRange(hsv_red, lower_red, upper_red)
        mask_red_clean = morphology(mask_red, red_ksize, red_open_iter, red_close_iter)
        
        # === XỬ LÝ MÀU VÀNG ===
        hsv_yellow = preprocess_frame(frame_to_process, yellow_clahe_clip_limit, yellow_blur_ksize)
        lower_yellow = np.array([yellow_lower_h, yellow_lower_s, yellow_lower_v])
        upper_yellow = np.array([yellow_upper_h, yellow_upper_s, yellow_upper_v])
        mask_yellow = cv2.inRange(hsv_yellow, lower_yellow, upper_yellow)
        mask_yellow_clean = morphology(mask_yellow, yellow_ksize, yellow_open_iter, yellow_close_iter)
        
        
        blue_roi_params = (
            int(width_new * 0.45), # roi_x_start
            0,                     # roi_y_start
            width_new,             # roi_x_end
            height_new             # roi_y_end
        )
        
        red_roi_params = (
            int(width_new * 0.45), # roi_x_start
            0,                     # roi_y_start
            width_new,             # roi_x_end
            int(height_new*0.94)   # roi_y_end
        )
        
        yellow_roi_params = (
            int(width_new * 0.45), # roi_x_start
            int(height_new * 0.5), # roi_y_start
            width_new,             # roi_x_end
            int(height_new * 0.9)  # roi_y_end
        )
        
        all_detections = []
        all_detections.extend(extract_circle_detections(mask_blue_clean, blue_roi_params, 'blue'))
        all_detections.extend(extract_circle_detections(mask_red_clean, red_roi_params, 'red'))
        all_detections.extend(extract_triangle_detections(mask_yellow_clean, yellow_roi_params, 'yellow'))
        
        # Thêm vào temporal filter
        temporal_filter.add_detections(frame_count - 1, all_detections)
    
    cap.release()
    end_pass1 = time.time()
    
    # Thống kê Pass 1
    total_tracks, valid_tracks = temporal_filter.get_statistics()
    print(f"\n✅ PASS 1 hoàn thành trong {end_pass1 - start_pass1:.2f}s")
    print(f"📊 Thống kê:")
    print(f"   • Tổng số tracks phát hiện: {total_tracks}")
    print(f"   • Tracks hợp lệ (≥{MIN_DURATION_SEC}s): {valid_tracks}")
    print(f"   • Tracks bị loại (noise): {total_tracks - valid_tracks}")
    
    # =========================================================================
    # === PASS 2: VẼ LẠI VIDEO VỚI CHỈ VALIDATED DETECTIONS ===
    # =========================================================================
    
    print(f"\n🎨 PASS 2: Tạo video với temporal filtering...")
    start_pass2 = time.time()
    
    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(final_output, fourcc, fps, (w_orig, h_orig))
    
    if not video_writer.isOpened():
        print(f"❌ Lỗi: Không thể tạo file output '{final_output}'")
        cap.release()
        return
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame_full = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"   🎬 Đã render {frame_count}/{total_frames} frames...")
        
        # Lấy validated detections cho frame này
        validated = temporal_filter.get_validated_detections(frame_count - 1)
        
        # Chỉ vẽ các detections đã lọc, KHÔNG VẼ ROI
        frame_output = draw_detections(frame_full.copy(), validated)
        
        # Lưu frame
        video_writer.write(frame_output)
    
    cap.release()
    video_writer.release()
    end_pass2 = time.time()
    
    # =========================================================================
    # === KẾT THÚC ===
    # =========================================================================
    
    total_time = end_pass2 - start_pass1
    
    print("\n" + "=" * 60)
    print("✅  XỬ LÝ HOÀN TẤT!")
    print("=" * 60)
    print(f"📁 File output: {final_output}")
    print(f"⏱️  Thời gian Pass 1: {end_pass1 - start_pass1:.2f}s")
    print(f"⏱️  Thời gian Pass 2: {end_pass2 - start_pass2:.2f}s")
    print(f"⏱️  Tổng thời gian: {total_time:.2f}s")
    
    if total_tracks > 0: # Tránh lỗi chia cho 0
        print(f"\n📈 Hiệu quả lọc nhiễu:")
        print(f"   • Đã loại bỏ {total_tracks - valid_tracks}/{total_tracks} tracks")
        print(f"   • Tỷ lệ giữ lại: {valid_tracks/total_tracks*100:.1f}%")
    else:
        print("\n📈 Không phát hiện được track nào.")
        
    print("=" * 60)


if __name__ == "__main__":
    main()