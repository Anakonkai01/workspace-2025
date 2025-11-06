import cv2
import numpy as np

print("Đang tải ảnh...")
# --- 1. Tải ảnh ---
# Tải ảnh mẫu (template) và ảnh cảnh (scene)
img_template = cv2.imread('template.jpeg')
img_scene = cv2.imread('template.jpeg')

if img_template is None or img_scene is None:
    print("Lỗi: Không thể tải ảnh 'template.jpg' hoặc 'scene.jpg'.")
    print("Hãy đảm bảo 2 file ảnh nằm cùng thư mục với code.")
else:
    # Chuyển sang ảnh xám để xử lý
    img_template_gray = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
    img_scene_gray = cv2.cvtColor(img_scene, cv2.COLOR_BGR2GRAY)

    # --- 2. Khởi tạo ORB Detector ---
    # nfeatures: Số lượng đặc trưng tối đa cần tìm
    orb = cv2.ORB_create(nfeatures=20000)

    # --- 3. Tính Keypoints & Descriptors ---
    # Tính cho ảnh mẫu
    (kp_template, des_template) = orb.detectAndCompute(img_template_gray, None)
    # Tính cho ảnh cảnh
    (kp_scene, des_scene) = orb.detectAndCompute(img_scene_gray, None)

    if des_template is None or des_scene is None:
        print("Không tìm thấy đủ keypoint trong một trong hai ảnh.")
    else:
        # --- 4. Đối sánh (Matching) ---
        # Sử dụng BFMatcher (Brute-Force) với NORM_HAMMING (phù hợp cho ORB)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        # Tìm 2 khớp tốt nhất (k=2) cho mỗi descriptor
        matches = bf.knnMatch(des_template, des_scene, k=2)
        
        # --- 5. Lọc (Lowe's Ratio Test) ---
        good_matches = []
        try:
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        except ValueError:
            print("Lỗi: Không đủ 'matches' để thực hiện Ratio Test.")

        # --- 6. BƯỚC PHÂN LOẠI (CLASSIFICATION) ---
        MIN_MATCH_COUNT = 10  # Ngưỡng: Cần ít nhất 10 điểm khớp "tốt"
        
        print(f"Đã tìm thấy {len(good_matches)} điểm khớp tốt.")

        if len(good_matches) > MIN_MATCH_COUNT:
            print(f"PHÂN LOẠI: TÌM THẤY BIỂN BÁO ( {len(good_matches)} điểm khớp).")

            # --- 7. BƯỚC TRÍCH XUẤT (EXTRACTION) ---
            
            # 7a. Lấy tọa độ các điểm khớp tốt
            src_pts = np.float32([ kp_template[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp_scene[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
            
            # 7b. Tìm Homography (Ma trận biến đổi)
            # M là ma trận 3x3 mô tả phép biến đổi (xoay, nghiêng,...)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # 7c. [Trực quan] Vẽ khung bao quanh biển báo trong ảnh cảnh
            h, w = img_template_gray.shape
            # Lấy 4 góc của ảnh template
            pts = np.float32([ [0,0], [0,h-1], [w-1,h-1], [w-1,0] ]).reshape(-1,1,2)
            # Biến đổi 4 góc đó sang hệ tọa độ của ảnh cảnh
            dst_box = cv2.perspectiveTransform(pts, M)
            
            # Vẽ khung lên ảnh cảnh (tạo bản sao để vẽ)
            img_scene_with_box = img_scene.copy()
            cv2.polylines(img_scene_with_box, [np.int32(dst_box)], True, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow("1. Tìm thấy biển báo", img_scene_with_box)

            # 7d. [Trích xuất] Làm phẳng ảnh (Warping)
            h_template, w_template = img_template.shape[:2]
            # Dùng ma trận M để "bẻ" (warp) ảnh cảnh
            # Kích thước đầu ra (w_template, h_template) bằng kích thước ảnh mẫu
            img_warped = cv2.warpPerspective(img_scene, M, (w_template, h_template))
            
            print("\nĐã trích xuất nội dung (làm phẳng).")
            cv2.imshow("2. Trích xuất (Làm phẳng)", img_warped)
            
            # (Tùy chọn) Lưu file trích xuất
            # cv2.imwrite("trich_xuat.jpg", img_warped)

            cv2.waitKey(0)

        else:
            print(f"PHÂN LOẠI: KHÔNG TÌM THẤY BIỂN BÁO (Chỉ có {len(good_matches)} điểm khớp).")

    cv2.destroyAllWindows()