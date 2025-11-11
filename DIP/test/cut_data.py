# import cv2
# import os
# import time
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from multiprocessing import freeze_support  # Cần thiết cho Windows
#
# # --- Import các thành phần cần thiết từ file gốc (task1.py) ---
# # (Thay '523H0164_523H0177_523H0145_task1' bằng tên file gốc của bạn nếu khác)
# try:
#     from task1 import(
#         # Classes
#         TemporalSignFilter,
#
#         # Functions
#         process_frame_batch,
#
#         # Global Variables (Parameters)
#         INPUT_VIDEO_PATH, MAX_FRAME_ID, BATCH_SIZE, NUM_PROCESS_WORKERS,
#         BLUE_MIN_DURATION_SEC, BLUE_MAX_GAP_SEC, BLUE_IOU_THRESHOLD,
#         RED_MIN_DURATION_SEC, RED_MAX_GAP_SEC, RED_IOU_THRESHOLD,
#         YELLOW_MIN_DURATION_SEC, YELLOW_MAX_GAP_SEC, YELLOW_IOU_THRESHOLD,
#         CIRCLE_MIN_AREA, CIRCLE_MAX_AREA, TRIANGLE_MIN_AREA, TRIANGLE_MAX_AREA
#     )
# except ImportError:
#     print("LỖI: Không tìm thấy file 'task1.py'.")
#     print("Hãy đảm bảo file này ở cùng thư mục và không bị đổi tên.")
#     exit()
# except SyntaxError:
#     print("LỖI: Tên file gốc (bắt đầu bằng số) không thể import trực tiếp.")
#     print("Vui lòng đổi tên file gốc thành 'task1_logic.py' và đổi tên dòng import trong file này.")
#     exit()
#
# # --- Các tham số riêng cho file cropper này ---
# CROP_OUTPUT_DIR = "sign_crops_test"  # Tên thư mục mới để tránh trùng lặp
# CROP_INTERVAL_SEC = 3.0  # Cắt mỗi 3 giây
# CROP_PADDING = 0  # Padding 10px
#
#
# def run_detection_phase(cap, fps):
#     """
#     Hàm này sao chép Giai đoạn Detection từ main() của file gốc
#     để xây dựng đối tượng temporal_filter.
#     """
#     print("PHASE 1: DETECTION (Đang chạy để xây dựng bộ lọc...)")
#     start_detection = time.time()
#
#     w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#     # Tạo dictionary thông số màu
#     color_params = {
#         'blue': (BLUE_MIN_DURATION_SEC, BLUE_MAX_GAP_SEC, BLUE_IOU_THRESHOLD),
#         'red': (RED_MIN_DURATION_SEC, RED_MAX_GAP_SEC, RED_IOU_THRESHOLD),
#         'yellow': (YELLOW_MIN_DURATION_SEC, YELLOW_MAX_GAP_SEC, YELLOW_IOU_THRESHOLD)
#     }
#
#     # Khởi tạo bộ lọc (import từ file gốc)
#     temporal_filter = TemporalSignFilter(fps, color_params=color_params)
#
#     # Cấu hình crop (giống file gốc)
#     height_new = int(h_orig * 0.475)
#     width_new = w_orig
#
#     # Đọc các frame (giống file gốc)
#     all_frames = []
#     frame_count = 0
#     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Đảm bảo đọc từ đầu
#
#     print(f"   Đang tải {MAX_FRAME_ID} frames để phân tích...")
#     while cap.isOpened() and frame_count < MAX_FRAME_ID:
#         ret, frame_full = cap.read()
#         if not ret:
#             break
#         all_frames.append((frame_count, frame_full))
#         frame_count += 1
#
#     print(f"   Đã tải {len(all_frames)} frames.")
#
#     # Chuẩn bị batch (giống file gốc)
#     batches = []
#     for i in range(0, len(all_frames), BATCH_SIZE):
#         batch = all_frames[i:i + BATCH_SIZE]
#         batches.append((batch, height_new, width_new, w_orig, h_orig,
#                         CIRCLE_MIN_AREA, CIRCLE_MAX_AREA, TRIANGLE_MIN_AREA, TRIANGLE_MAX_AREA))
#
#     # Chạy xử lý đa tiến trình (giống file gốc)
#     print(f"   Đang xử lý với {NUM_PROCESS_WORKERS} workers...")
#     all_detections_results = []
#
#     with ProcessPoolExecutor(max_workers=NUM_PROCESS_WORKERS) as executor:
#         # Sử dụng hàm process_frame_batch (import từ file gốc)
#         futures = {executor.submit(process_frame_batch, batch): i for i, batch in enumerate(batches)}
#
#         completed = 0
#         for future in as_completed(futures):
#             batch_results = future.result()
#             all_detections_results.extend(batch_results)
#             completed += 1
#
#             if completed % 10 == 0 or completed == len(batches):
#                 progress = int(completed / len(batches) * 100)
#                 print(f"   Đã xử lý {completed}/{len(batches)} batches ({progress}%)")
#
#     # Nạp phát hiện vào bộ lọc (giống file gốc)
#     all_detections_results.sort(key=lambda x: x[0])
#
#     print("   Đang xây dựng temporal tracks...")
#     for frame_num, detections in all_detections_results:
#         temporal_filter.add_detections(frame_num, detections)
#
#     print(f"PHASE 1 hoàn thành trong {time.time() - start_detection:.2f}s")
#
#     # Quan trọng: Xây dựng cache
#     temporal_filter.build_detection_cache()
#
#     return temporal_filter
#
#
# def run_cropping_phase(cap, fps, temporal_filter):
#     """
#     Hàm này chạy Giai đoạn Cắt ảnh (Cropping),
#     sử dụng temporal_filter đã được xây dựng từ Phase 1.
#     """
#     print("\nPHASE 2: CROPPING (Đang đọc video và lưu ảnh cắt...)")
#
#     if not os.path.exists(CROP_OUTPUT_DIR):
#         os.makedirs(CROP_OUTPUT_DIR)
#         print(f"   Đã tạo thư mục: {CROP_OUTPUT_DIR}")
#
#     # --- ĐÃ XÓA LOGIC interval_frames ---
#
#     print(f"   Sẽ cắt TẤT CẢ các biển báo hợp lệ được tìm thấy trên MỌI FRAME.")
#
#     frame_num = 0
#     frames_saved_count = 0
#
#     # Đảm bảo đọc lại video từ đầu
#     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break  # Kết thúc video
#
#         # --- ĐÃ XÓA if frame_num % interval_frames == 0: ---
#         # Logic này giờ sẽ chạy trên MỌI frame
#
#         # Lấy các phát hiện ĐÃ ĐƯỢC LỌC từ cache của bộ lọc
#         validated_detections = temporal_filter.get_validated_detections(frame_num)
#
#         if validated_detections:
#             # (Tùy chọn: Bạn có thể xóa dòng print bên dưới nếu nó xuất hiện quá nhiều)
#             # print(f"   -> Tìm thấy {len(validated_detections)} biển báo tại frame {frame_num}.")
#
#             for i, detection_data in enumerate(validated_detections):
#                 # detection_data là (bbox, color, metrics)
#                 bbox = detection_data[0]
#                 color_type = detection_data[1]
#
#                 x, y, w, h = bbox
#
#                 # Cắt biển báo (thêm padding)
#                 y_start = max(0, y - CROP_PADDING)
#                 y_end = min(frame.shape[0], y + h + CROP_PADDING)
#                 x_start = max(0, x - CROP_PADDING)
#                 x_end = min(frame.shape[1], x + w + CROP_PADDING)
#
#                 # Chỉ cắt nếu tọa độ hợp lệ
#                 if y_start < y_end and x_start < x_end:
#                     cropped_sign = frame[y_start:y_end, x_start:x_end]
#
#                     # Tạo tên file và lưu
#                     filename = os.path.join(CROP_OUTPUT_DIR, f"frame{frame_num}_{color_type}_id{i}.jpg")
#                     cv2.imwrite(filename, cropped_sign)
#                     frames_saved_count += 1
#
#         frame_num += 1
#
#     print(f"PHASE 2 hoàn thành. Đã lưu tổng cộng {frames_saved_count} ảnh.")
#
#
# def main_cropper():
#     """
#     Hàm main() của file cropper.
#     """
#     print("=" * 70)
#     print("CHƯƠNG TRÌNH CẮT ẢNH BIỂN BÁO (TÁI SỬ DỤNG CODE TỪ TASK 1)")
#     print("=" * 70)
#
#     cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
#     if not cap.isOpened():
#         print(f"LỖI: Không thể mở video '{INPUT_VIDEO_PATH}'")
#         return
#
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     if fps == 0:
#         fps = 30.0  # Giá trị mặc định nếu không đọc được fps
#
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print(f"Video: {INPUT_VIDEO_PATH}")
#     print(f"Tổng số frame: {total_frames}, FPS: {fps:.2f}")
#
#     # --- Bước 1: Chạy Detection Phase để lấy filter ---
#     # (Việc này sẽ tốn thời gian, vì nó phải xử lý các frame)
#     try:
#         filter_data = run_detection_phase(cap, fps)
#     except Exception as e:
#         print(f"LỖI trong Giai đoạn Detection: {e}")
#         cap.release()
#         return
#
#     # --- Bước 2: Chạy Cropping Phase ---
#     # (Đọc lại video và dùng filter để cắt ảnh)
#     try:
#         run_cropping_phase(cap, fps, filter_data)
#     except Exception as e:
#         print(f"LỖI trong Giai đoạn Cropping: {e}")
#
#     cap.release()
#     print("=" * 70)
#     print("Hoàn thành.")
#     print(f"Ảnh đã được lưu tại thư mục: {CROP_OUTPUT_DIR}")
#     print("=" * 70)
#
#
# if __name__ == "__main__":
#     # Cần thiết để ProcessPoolExecutor hoạt động ổn định
#     # khi chạy file script (đặc biệt trên Windows)
#     freeze_support()
#
#     main_cropper()

import cv2
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support  # Cần thiết cho Windows

# --- Import các thành phần cần thiết từ file gốc (task1.py) ---
# (Thay '523H0164_523H0177_523H0145_task1' bằng tên file gốc của bạn nếu khác)
try:
    from task1 import(
        # Classes
        TemporalSignFilter,

        # Functions
        process_frame_batch,

        # Global Variables (Parameters)
        INPUT_VIDEO_PATH, MAX_FRAME_ID, BATCH_SIZE, NUM_PROCESS_WORKERS,
        BLUE_MIN_DURATION_SEC, BLUE_MAX_GAP_SEC, BLUE_IOU_THRESHOLD,
        RED_MIN_DURATION_SEC, RED_MAX_GAP_SEC, RED_IOU_THRESHOLD,
        YELLOW_MIN_DURATION_SEC, YELLOW_MAX_GAP_SEC, YELLOW_IOU_THRESHOLD,
        CIRCLE_MIN_AREA, CIRCLE_MAX_AREA, TRIANGLE_MIN_AREA, TRIANGLE_MAX_AREA
    )
except ImportError:
    print("LỖI: Không tìm thấy file '523H0164_523H0177_523H0145_task1.py'.")
    print("Hãy đảm bảo file này ở cùng thư mục và không bị đổi tên.")
    exit()
except SyntaxError:
    print("LỖI: Tên file gốc (bắt đầu bằng số) không thể import trực tiếp.")
    print("Vui lòng đổi tên file gốc thành 'task1_logic.py' và đổi tên dòng import trong file này.")
    exit()

# --- Các tham số riêng cho file cropper này ---
CROP_OUTPUT_DIR = "sign_crops_new_2"  # Tên thư mục mới để tránh trùng lặp
CROP_INTERVAL_SEC = 3.0  # Cắt mỗi 3 giây
CROP_PADDING = 10  # Padding 10px


def run_detection_phase(cap, fps):
    """
    Hàm này sao chép Giai đoạn Detection từ main() của file gốc
    để xây dựng đối tượng temporal_filter.
    """
    print("PHASE 1: DETECTION (Đang chạy để xây dựng bộ lọc...)")
    start_detection = time.time()

    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Tạo dictionary thông số màu
    color_params = {
        'blue': (BLUE_MIN_DURATION_SEC, BLUE_MAX_GAP_SEC, BLUE_IOU_THRESHOLD),
        'red': (RED_MIN_DURATION_SEC, RED_MAX_GAP_SEC, RED_IOU_THRESHOLD),
        'yellow': (YELLOW_MIN_DURATION_SEC, YELLOW_MAX_GAP_SEC, YELLOW_IOU_THRESHOLD)
    }

    # Khởi tạo bộ lọc (import từ file gốc)
    temporal_filter = TemporalSignFilter(fps, color_params=color_params)

    # Cấu hình crop (giống file gốc)
    height_new = int(h_orig * 0.475)
    width_new = w_orig

    # Đọc các frame (giống file gốc)
    all_frames = []
    frame_count = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Đảm bảo đọc từ đầu

    print(f"   Đang tải {MAX_FRAME_ID} frames để phân tích...")
    while cap.isOpened() and frame_count < MAX_FRAME_ID:
        ret, frame_full = cap.read()
        if not ret:
            break
        all_frames.append((frame_count, frame_full))
        frame_count += 1

    print(f"   Đã tải {len(all_frames)} frames.")

    # Chuẩn bị batch (giống file gốc)
    batches = []
    for i in range(0, len(all_frames), BATCH_SIZE):
        batch = all_frames[i:i + BATCH_SIZE]
        batches.append((batch, height_new, width_new, w_orig, h_orig,
                        CIRCLE_MIN_AREA, CIRCLE_MAX_AREA, TRIANGLE_MIN_AREA, TRIANGLE_MAX_AREA))

    # Chạy xử lý đa tiến trình (giống file gốc)
    print(f"   Đang xử lý với {NUM_PROCESS_WORKERS} workers...")
    all_detections_results = []

    with ProcessPoolExecutor(max_workers=NUM_PROCESS_WORKERS) as executor:
        # Sử dụng hàm process_frame_batch (import từ file gốc)
        futures = {executor.submit(process_frame_batch, batch): i for i, batch in enumerate(batches)}

        completed = 0
        for future in as_completed(futures):
            batch_results = future.result()
            all_detections_results.extend(batch_results)
            completed += 1

            if completed % 10 == 0 or completed == len(batches):
                progress = int(completed / len(batches) * 100)
                print(f"   Đã xử lý {completed}/{len(batches)} batches ({progress}%)")

    # Nạp phát hiện vào bộ lọc (giống file gốc)
    all_detections_results.sort(key=lambda x: x[0])

    print("   Đang xây dựng temporal tracks...")
    for frame_num, detections in all_detections_results:
        temporal_filter.add_detections(frame_num, detections)

    print(f"PHASE 1 hoàn thành trong {time.time() - start_detection:.2f}s")

    # Quan trọng: Xây dựng cache
    temporal_filter.build_detection_cache()

    return temporal_filter


def run_cropping_phase(cap, fps, temporal_filter):
    """
    Hàm này chạy Giai đoạn Cắt ảnh (Cropping),
    sử dụng temporal_filter đã được xây dựng từ Phase 1.
    """
    print("\nPHASE 2: CROPPING (Đang đọc video và lưu ảnh cắt...)")

    if not os.path.exists(CROP_OUTPUT_DIR):
        os.makedirs(CROP_OUTPUT_DIR)
        print(f"   Đã tạo thư mục: {CROP_OUTPUT_DIR}")

    # Tính toán số frame cho mỗi 3 giây
    interval_frames = int(CROP_INTERVAL_SEC * fps)

    if interval_frames <= 0:
        print(f"LỖI: Khoảng cách frame là 0 (FPS={fps}). Không thể tiếp tục.")
        return

    print(f"   FPS={fps:.2f}, sẽ cắt ảnh mỗi {interval_frames} frames (mỗi {CROP_INTERVAL_SEC} giây).")

    frame_num = 0
    frames_saved_count = 0

    # Đảm bảo đọc lại video từ đầu
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Kết thúc video

        # Chỉ kiểm tra các frame nằm ở mốc 3 giây
        if frame_num % interval_frames == 0:

            # Lấy các phát hiện ĐÃ ĐƯỢC LỌC từ cache của bộ lọc
            validated_detections = temporal_filter.get_validated_detections(frame_num)

            if validated_detections:
                print(f"   -> Tìm thấy {len(validated_detections)} biển báo tại frame {frame_num}.")

                for i, detection_data in enumerate(validated_detections):
                    # detection_data là (bbox, color, metrics)
                    bbox = detection_data[0]
                    color_type = detection_data[1]

                    x, y, w, h = bbox

                    # Cắt biển báo (thêm padding)
                    y_start = max(0, y - CROP_PADDING)
                    y_end = min(frame.shape[0], y + h + CROP_PADDING)
                    x_start = max(0, x - CROP_PADDING)
                    x_end = min(frame.shape[1], x + w + CROP_PADDING)

                    # Chỉ cắt nếu tọa độ hợp lệ
                    if y_start < y_end and x_start < x_end:
                        cropped_sign = frame[y_start:y_end, x_start:x_end]

                        # Tạo tên file và lưu
                        filename = os.path.join(CROP_OUTPUT_DIR, f"frame{frame_num}_{color_type}_id{i}.jpg")
                        cv2.imwrite(filename, cropped_sign)
                        frames_saved_count += 1

        frame_num += 1

    print(f"PHASE 2 hoàn thành. Đã lưu tổng cộng {frames_saved_count} ảnh.")


def main_cropper():
    """
    Hàm main() của file cropper.
    """
    print("=" * 70)
    print("CHƯƠNG TRÌNH CẮT ẢNH BIỂN BÁO (TÁI SỬ DỤNG CODE TỪ TASK 1)")
    print("=" * 70)

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"LỖI: Không thể mở video '{INPUT_VIDEO_PATH}'")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0  # Giá trị mặc định nếu không đọc được fps

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {INPUT_VIDEO_PATH}")
    print(f"Tổng số frame: {total_frames}, FPS: {fps:.2f}")

    # --- Bước 1: Chạy Detection Phase để lấy filter ---
    # (Việc này sẽ tốn thời gian, vì nó phải xử lý các frame)
    try:
        filter_data = run_detection_phase(cap, fps)
    except Exception as e:
        print(f"LỖI trong Giai đoạn Detection: {e}")
        cap.release()
        return

    # --- Bước 2: Chạy Cropping Phase ---
    # (Đọc lại video và dùng filter để cắt ảnh)
    try:
        run_cropping_phase(cap, fps, filter_data)
    except Exception as e:
        print(f"LỖI trong Giai đoạn Cropping: {e}")

    cap.release()
    print("=" * 70)
    print("Hoàn thành.")
    print(f"Ảnh đã được lưu tại thư mục: {CROP_OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    # Cần thiết để ProcessPoolExecutor hoạt động ổn định
    # khi chạy file script (đặc biệt trên Windows)
    freeze_support()

    main_cropper()