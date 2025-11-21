import cv2
import numpy as np
import os
import glob
import shutil
from typing import Optional

# =============================================================================
# CÀI ĐẶT
# =============================================================================
CANDIDATES_DIR = "temp_candidates" 
MODEL_PATH = "svm_sign_detector_v3.xml" # Model "tốt" của bạn
OUTPUT_DIR = "temp_sorted"

OUTPUT_POSITIVES = os.path.join(OUTPUT_DIR, "positives")
OUTPUT_NEGATIVES = os.path.join(OUTPUT_DIR, "negatives")

# Cấu hình HOG (PHẢI GIỐNG HỆT KHI TRAIN)
RESIZE_DIM = (64, 64)
winSize = RESIZE_DIM
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

# =============================================================================
# HÀM TRỢ GIÚP
# =============================================================================
def get_hog_features_from_image(img_gray: np.ndarray) -> Optional[np.ndarray]:
    try:
        img_resized = cv2.resize(img_gray, RESIZE_DIM, interpolation=cv2.INTER_AREA)
        features = hog.compute(img_resized)
        return features.flatten()
    except Exception:
        return None

# =============================================================================
# HÀM CHÍNH: PHÂN LOẠI
# =============================================================================
def sort_all_candidates():
    print("="*70)
    print("  BẮT ĐẦU PHÂN LOẠI 55,000 ỨNG VIÊN BẰNG MODEL V1")
    print("="*70)
    
    print(f"Đang tải Model từ: {MODEL_PATH}")
    try:
        svm = cv2.ml.SVM_load(MODEL_PATH)
        if svm is None: raise Exception("Model is None")
    except Exception as e:
        print(f"Lỗi nghiêm trọng: Không thể tải model '{MODEL_PATH}'. {e}")
        return

    os.makedirs(OUTPUT_POSITIVES, exist_ok=True)
    os.makedirs(OUTPUT_NEGATIVES, exist_ok=True)
    
    print(f"Đang quét thư mục ứng viên: {CANDIDATES_DIR}")
    all_candidates = glob.glob(os.path.join(CANDIDATES_DIR, "*.*"))
    total = len(all_candidates)
    
    if total == 0:
        print("Lỗi: Không tìm thấy ứng viên.")
        return
        
    print(f"Tìm thấy {total} ứng viên. Bắt đầu phân loại...")
    
    pos_count = 0
    neg_count = 0
    
    for i, img_path in enumerate(all_candidates):
        try:
            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None: continue
            
            features = get_hog_features_from_image(img_gray)
            if features is None: continue
            
            # Dự đoán
            _, result = svm.predict(features.reshape(1, -1))
            prediction = int(result[0][0])
            
            if prediction == 1:
                # Model nghĩ đây là biển báo
                shutil.copy(img_path, os.path.join(OUTPUT_POSITIVES, os.path.basename(img_path)))
                pos_count += 1
            else:
                # Model nghĩ đây KHÔNG phải biển báo
                shutil.copy(img_path, os.path.join(OUTPUT_NEGATIVES, os.path.basename(img_path)))
                neg_count += 1
                
        except Exception:
            pass
            
        if (i + 1) % 1000 == 0:
            print(f"  ... Đã xử lý {i+1} / {total} (Pos: {pos_count}, Neg: {neg_count})")

    print("\n" + "="*70)
    print("✓ Phân loại hoàn tất!")
    print(f"  Đã lưu {pos_count} ảnh vào: {OUTPUT_POSITIVES}")
    print(f"  Đã lưu {neg_count} ảnh vào: {OUTPUT_NEGATIVES}")
    print("="*70)

if __name__ == "__main__":
    sort_all_candidates()