import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import sys  # Dùng để thoát nếu có lỗi
import joblib

# --- 1. Các hằng số và cài đặt ---
# !!! THAY ĐỔI ĐƯỜNG DẪN ẢNH TEST CỦA BẠN Ở ĐÂY
TRAIN_DIR = 'sign_crops_new'  # Thư mục chứa dữ liệu train
TEST_IMAGE_PATH = 'sign_crops_test/frame2488_yellow_id0.jpg'  # <<< SỬA ĐƯỜNG DẪN NÀY

IMAGE_SIZE = (64, 64)  # Kích thước chuẩn để resize tất cả ảnh

# Các tham số HOG (phải giống hệt nhau cho cả train và test)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)


# --- 2. Hàm tải dữ liệu và trích xuất đặc trưng HOG ---
def load_data_and_extract_features(data_dir):
    features = []
    labels = []

    class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    if not class_names:
        print(f"LỖI: Không tìm thấy thư mục con nào trong: {data_dir}")
        return None, None

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            image = cv2.imread(img_path)
            if image is None:
                continue

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray_image, IMAGE_SIZE)

            # Trích xuất đặc trưng HOG
            feature_vector = hog(resized_image,
                                 orientations=HOG_ORIENTATIONS,
                                 pixels_per_cell=HOG_PIXELS_PER_CELL,
                                 cells_per_block=HOG_CELLS_PER_BLOCK,
                                 visualize=False,
                                 transform_sqrt=True)

            features.append(feature_vector)
            labels.append(class_name)

    return np.array(features), np.array(labels)


# --- 3. Tải dữ liệu TRAIN ---
print(f"Đang tải dữ liệu TRAIN từ: {TRAIN_DIR}...")
X_train, y_train = load_data_and_extract_features(TRAIN_DIR)

if X_train is None:
    print("Không thể tải dữ liệu train. Dừng chương trình.")
    sys.exit()

print(f"Hoàn thành! Tổng số ảnh train: {len(X_train)}")
print(f"Kích thước vector đặc trưng HOG: {X_train.shape[1]}")

# --- 4. Mã hóa nhãn (Labels) ---
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

print("\nÁnh xạ nhãn (học từ tập train):")
for i, class_name in enumerate(le.classes_):
    print(f"{i}: {class_name}")

# --- 5. Huấn luyện mô hình SVM ---
print("\nBắt đầu huấn luyện mô hình SVM trên toàn bộ tập train...")
model = SVC(kernel='linear', C=1.0, random_state=42, probability=True)
model.fit(X_train, y_train_encoded)
print("Huấn luyện hoàn tất!")
model_filename = 'svm_traffic_sign.joblib'
encoder_filename = 'label_encoder.joblib'

print(f"\nĐang lưu mô hình vào: {model_filename}")
joblib.dump(model, model_filename)

print(f"Đang lưu encoder vào: {encoder_filename}")
joblib.dump(le, encoder_filename)

print("Lưu hoàn tất!")
# --- 6. Dự đoán trên một ảnh đơn lẻ ---
print(f"\n--- Đang dự đoán cho ảnh: {TEST_IMAGE_PATH} ---")

try:
    # 6.1. Đọc và tiền xử lý ảnh test
    test_image = cv2.imread(TEST_IMAGE_PATH)
    if test_image is None:
        raise FileNotFoundError(f"Không thể đọc ảnh từ: {TEST_IMAGE_PATH}")

    # 6.2. Áp dụng các bước tiền xử lý Y HỆT như lúc train
    gray_test = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    resized_test = cv2.resize(gray_test, IMAGE_SIZE)

    # 6.3. Trích xuất đặc trưng HOG Y HỆT như lúc train
    hog_features_test = hog(resized_test,
                            orientations=HOG_ORIENTATIONS,
                            pixels_per_cell=HOG_PIXELS_PER_CELL,
                            cells_per_block=HOG_CELLS_PER_BLOCK,
                            visualize=False,
                            transform_sqrt=True)

    # 6.4. Reshape lại để phù hợp với input của SVM
    # model.predict() mong đợi một mảng 2D (danh sách các mẫu)
    # Chúng ta biến đổi vector [1, 2, 3] thành [[1, 2, 3]]
    hog_features_test = hog_features_test.reshape(1, -1)

    # 6.5. Thực hiện dự đoán
    prediction_encoded = model.predict(hog_features_test)
    prediction_proba = model.predict_proba(hog_features_test)

    # 6.6. Giải mã kết quả
    prediction_name = le.inverse_transform(prediction_encoded)
    confidence = np.max(prediction_proba) * 100

    print("\n========== KẾT QUẢ DỰ ĐOÁN ==========")
    print(f"Biển báo được nhận diện là: **{prediction_name[0]}**")
    print(f"Độ tự tin (Confidence): {confidence:.2f}%")
    print("========================================")

except FileNotFoundError as e:
    print(f"\nLỖI: {e}")
    print("Vui lòng kiểm tra lại biến 'TEST_IMAGE_PATH' ở đầu script.")
except Exception as e:
    print(f"\nĐã xảy ra lỗi không mong muốn: {e}")