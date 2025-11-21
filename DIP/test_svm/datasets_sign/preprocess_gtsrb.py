import os
import cv2
import csv
import shutil

# =============================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# =============================================================================
# Đường dẫn đến thư mục huấn luyện GTSRB bạn đã giải nén
# (Nó phải chứa các thư mục 00000, 00001, v.v.)
GTSRB_TRAIN_DIR = "GTSRB/Final_Training_Images/GTSRB/Training" 

# Thư mục đầu ra cho Model A (Recognizer)
RECOGNIZER_OUTPUT_DIR = "dataset_recognizer"

# Thư mục đầu ra cho Model B (Detector)
DETECTOR_OUTPUT_DIR = "dataset_detector"
DETECTOR_POSITIVE_DIR = os.path.join(DETECTOR_OUTPUT_DIR, "positive")

# Số lượng lớp (thư mục) trong GTSRB
NUM_CLASSES = 43 
IMG_SIZE = (64, 64) # Kích thước chuẩn để lưu lại (giống HOG)

# =============================================================================
# HÀM XỬ LÝ
# =============================================================================
def preprocess_gtsrb():
    print("Bắt đầu xử lý GTSRB...")
    
    # Tạo các thư mục đầu ra nếu chưa có
    os.makedirs(RECOGNIZER_OUTPUT_DIR, exist_ok=True)
    os.makedirs(DETECTOR_POSITIVE_DIR, exist_ok=True)
    
    total_images_processed = 0

    # Duyệt qua tất cả 43 lớp
    for class_id in range(NUM_CLASSES):
        class_str = f"{class_id:05d}" # Format: 00000, 00001, ...
        class_path = os.path.join(GTSRB_TRAIN_DIR, class_str)
        
        if not os.path.isdir(class_path):
            print(f"Warning: Không tìm thấy thư mục {class_path}")
            continue
            
        # 1. Chuẩn bị thư mục cho Model A (Recognizer)
        # Chúng ta dùng tên dễ hiểu hơn, ví dụ: "Class_00000"
        recognizer_class_dir = os.path.join(RECOGNIZER_OUTPUT_DIR, f"Class_{class_str}")
        os.makedirs(recognizer_class_dir, exist_ok=True)
        
        # Mở file CSV chứa thông tin bounding box
        csv_path = os.path.join(class_path, f"GT-{class_str}.csv")
        if not os.path.isfile(csv_path):
            print(f"Warning: Không tìm thấy file CSV: {csv_path}")
            continue
            
        print(f"Đang xử lý Lớp: {class_str}...")
        
        with open(csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            next(reader) # Bỏ qua dòng tiêu đề (header)
            
            for row in reader:
                try:
                    filename = row[0]
                    img_path = os.path.join(class_path, filename)
                    
                    # Đọc ảnh gốc
                    img = cv2.imread(img_path)
                    if img is None: continue
                        
                    # Lấy tọa độ cắt
                    x1 = int(row[3])
                    y1 = int(row[4])
                    x2 = int(row[5])
                    y2 = int(row[6])
                    
                    # Cắt (crop) biển báo ra
                    cropped_img = img[y1:y2, x1:x2]
                    
                    # Resize về kích thước HOG chuẩn
                    resized_img = cv2.resize(cropped_img, IMG_SIZE, interpolation=cv2.INTER_AREA)
                    
                    # Tạo tên file output duy nhất
                    output_filename = f"{class_str}_{filename.replace('.ppm', '.png')}"
                    
                    # 2. Lưu cho Model A (Recognizer)
                    save_path_A = os.path.join(recognizer_class_dir, output_filename)
                    cv2.imwrite(save_path_A, resized_img)
                    
                    # 3. Lưu cho Model B (Detector - Positive)
                    save_path_B = os.path.join(DETECTOR_POSITIVE_DIR, output_filename)
                    cv2.imwrite(save_path_B, resized_img)
                    
                    total_images_processed += 1
                
                except Exception as e:
                    print(f"Lỗi khi xử lý {row[0]}: {e}")

    print("\n" + "="*50)
    print("✓ Xử lý GTSRB hoàn tất!")
    print(f"Tổng cộng {total_images_processed} ảnh đã được cắt và lưu.")
    print(f"  -> Dữ liệu Model A (Recognizer) tại: {RECOGNIZER_OUTPUT_DIR}")
    print(f"  -> Dữ liệu Model B (Detector) tại: {DETECTOR_POSITIVE_DIR}")
    print("="*50)

# =============================================================================
# CHẠY CHƯƠNG TRÌNH
# =============================================================================
if __name__ == "__main__":
    if not os.path.isdir(GTSRB_TRAIN_DIR):
        print(f"Lỗi: Không tìm thấy đường dẫn '{GTSRB_TRAIN_DIR}'")
        print("Hãy tải GTSRB, giải nén và cập nhật đường dẫn GTSRB_TRAIN_DIR.")
    else:
        preprocess_gtsrb()