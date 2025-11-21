import cv2
import numpy as np
import os
import glob
from typing import Optional, Dict

# =============================================================================
# CẤU HÌNH (BẮT BUỘC THAY ĐỔI)
# =============================================================================

# 1. ĐẶT TÊN CÁC FILE MODEL CỦA BẠN
MODEL_DETECTOR_PATH = "svm_sign_detector_v3.xml"  # Model B (Sign/Not-Sign)
MODEL_RECOGNIZER_PATH = "svm_sign_recognizer_v1.yml" # Model A (Multi-class)

# 2. TẠO LABEL MAP CỦA BẠN (TỪ KẾT QUẢ train_recognizer.py)
# !!! BẠN PHẢI TỰ ĐIỀN CÁI NÀY !!!
LABEL_MAP_CUSTOM = {
    0: "cam_di_nguoc_chieu",
    1: "cam_queo_trai",
    2: "cam_do_xe",
    3: "cam_dung_do_xe",
    4: "huong_ben_phai",
    5: "canh_bao_nguy_hiem"
    # (Thêm các nhãn khác của bạn vào đây...)
}

# 3. ĐẶT ĐƯỜNG DẪN ĐẾN ẢNH BẠN MUỐN TEST
# (Dùng * để test tất cả ảnh trong 1 thư mục)
TEST_IMAGE_PATHS = [
    "/home/hpenvy/workspace-2025/DIP/test_svm/datasets_sign/data_candidates_for_detector/positives/cand_frame0_blue_b6a4.png",  # <-- Test 1 ảnh Positive
    "/home/hpenvy/workspace-2025/DIP/test_svm/datasets_sign/data_candidates_for_detector/positives/cand_frame0_red_4a50.png",  # <-- Test 1 ảnh Negative
    "/home/hpenvy/workspace-2025/DIP/test_svm/datasets_sign/data_candidates_for_detector/negatives/cand_frame0_blue_7a8c.png", # <-- Test ảnh cụ thể
    "/home/hpenvy/workspace-2025/DIP/test_svm/datasets_sign/image.png" # <-- Test 1 frame từ video
]

# Hoặc, để test toàn bộ thư mục:
# TEST_IMAGE_PATHS = glob.glob("test_images_folder/*.png")


# =============================================================================
# CẤU HÌNH HOG (PHẢI GIỐNG HỆT KHI TRAIN)
# =============================================================================
RESIZE_DIM = (64, 64)
winSize = RESIZE_DIM
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

# =============================================================================
# HÀM TRÍCH XUẤT HOG
# =============================================================================
def get_hog_features_from_image(img_gray: np.ndarray) -> Optional[np.ndarray]:
    try:
        img_resized = cv2.resize(img_gray, RESIZE_DIM, interpolation=cv2.INTER_AREA)
        features = hog.compute(img_resized)
        return features.flatten()
    except Exception:
        return None

# =============================================================================
# HÀM TEST PIPELINE
# =============================================================================
def test_pipeline(image_path: str, svm_detector, svm_recognizer, label_map):
    print(f"\n--- Đang test ảnh: {os.path.basename(image_path)} ---")
    
    # 1. Tải và trích xuất HOG
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("  Lỗi: Không thể đọc ảnh.")
        return
        
    features = get_hog_features_from_image(img)
    if features is None:
        print("  Lỗi: Không thể trích xuất HOG.")
        return
        
    features_batch = features.reshape(1, -1) # Định dạng cho SVM
    
    # 2. Test Model B (Detector)
    _, result_detector = svm_detector.predict(features_batch)
    prediction_detector = int(result_detector[0][0])
    
    if prediction_detector == 0:
        print("  ✅ Model B (Detector) nói: [KHÔNG PHẢI BIỂN BÁO]")
        print("  (Pipeline dừng ở đây)")
        return

    # Nếu code chạy đến đây, nghĩa là Model B nói "Đây LÀ biển báo"
    print("  ✅ Model B (Detector) nói: [LÀ BIỂN BÁO]")
    
    # 3. Test Model A (Recognizer)
    _, result_recognizer = svm_recognizer.predict(features_batch)
    prediction_recognizer_id = int(result_recognizer[0][0])
    
    # Dịch ID sang Tên
    sign_name = label_map.get(prediction_recognizer_id, "Unknown ID")
    
    print(f"  ✅ Model A (Recognizer) nói: [ID: {prediction_recognizer_id}] => [TÊN: {sign_name}]")

# =============================================================================
# HÀM CHẠY CHÍNH
# =============================================================================
def main():
    print("="*70)
    print("  BẮT ĐẦU TEST PIPELINE HOG + SVM (B + A)")
    print("="*70)

    # 1. Tải Model B (Detector)
    try:
        svm_detector = cv2.ml.SVM_load(MODEL_DETECTOR_PATH)
        if svm_detector is None: raise Exception("Model is None")
        print(f"✓ Tải Model B ({MODEL_DETECTOR_PATH}) thành công.")
    except Exception as e:
        print(f"❌ Lỗi: Không thể tải Model B tại '{MODEL_DETECTOR_PATH}'. {e}")
        return

    # 2. Tải Model A (Recognizer)
    try:
        svm_recognizer = cv2.ml.SVM_load(MODEL_RECOGNIZER_PATH)
        if svm_recognizer is None: raise Exception("Model is None")
        print(f"✓ Tải Model A ({MODEL_RECOGNIZER_PATH}) thành công.")
    except Exception as e:
        print(f"❌ Lỗi: Không thể tải Model A tại '{MODEL_RECOGNIZER_PATH}'. {e}")
        return

    # 3. Kiểm tra Label Map
    if not LABEL_MAP_CUSTOM:
        print("\n❌ Lỗi: Bạn chưa điền 'LABEL_MAP_CUSTOM' ở đầu script.")
        print("  Hãy mở script và thêm map (ví dụ: {0: 'Stop', 1: 'Turn Left'}).")
        return
    
    # 4. Chạy test trên từng ảnh
    for img_path in TEST_IMAGE_PATHS:
        if not os.path.exists(img_path):
            print(f"\n--- Warning: Bỏ qua ảnh không tồn tại: {img_path} ---")
            continue
        test_pipeline(img_path, svm_detector, svm_recognizer, LABEL_MAP_CUSTOM)
        
    print("\n" + "="*70)
    print("✓ Test hoàn tất.")
    print("="*70)

if __name__ == "__main__":
    main()