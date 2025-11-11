# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # --- 1. Tải ảnh (Chỉ tải 1 LẦN DUY NHẤT) ---
# # Tải ảnh MÀU để hiển thị
# img_query_color = cv2.imread('./img/qc-2.jpg', cv2.IMREAD_COLOR)
# img_scene_color = cv2.imread('./img/img.png', cv2.IMREAD_COLOR)
#
# # Kiểm tra ngay lập tức xem ảnh có tải được không
# if img_query_color is None or img_scene_color is None:
#     print("Lỗi: Không thể tải ảnh. Hãy kiểm tra lại đường dẫn file.")
#     print("Script đang tìm ảnh tại thư mục:", __file__)
# else:
#     # Chuyển sang ảnh XÁM để xử lý
#     img_query = cv2.cvtColor(img_query_color, cv2.COLOR_BGR2GRAY)
#     img_scene = cv2.cvtColor(img_scene_color, cv2.COLOR_BGR2GRAY)
#
#     # --- 2. Khởi tạo ORB Detector ---
#     orb = cv2.ORB_create(nfeatures=1000)
#
#     # --- 3. Tìm Keypoints và Descriptors (trên ảnh xám) ---
#     kp_query, des_query = orb.detectAndCompute(img_query, None)
#     kp_scene, des_scene = orb.detectAndCompute(img_scene, None)
#
#     if des_query is None or des_scene is None:
#         print("Không tìm thấy đủ đặc trưng (descriptors) trong một trong hai ảnh.")
#     else:
#         # --- 4. Khởi tạo Brute-Force Matcher ---
#         bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
#         matches = bf.knnMatch(des_query, des_scene, k=2)
#
#         # --- 5. Lọc diêm khớp (Ratio Test) ---
#         good_matches = []
#         for item in matches:
#             if len(item) == 2:
#                 m, n = item
#                 if m.distance < 0.75 * n.distance:
#                     good_matches.append(m)
#
#         print(f"Tìm thấy {len(good_matches)} diêm khớp tốt.")
#
#         # Tạo một ảnh kết quả (ảnh màu) để vẽ lên
#         img_result = img_scene_color.copy()
#
#         # --- 6. Vẽ hình hộp định vị ---
#         MIN_MATCH_COUNT = 10
#
#         if len(good_matches) > MIN_MATCH_COUNT:
#             src_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#             dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#
#             M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#
#             if M is not None:
#                 h, w = img_query.shape
#                 pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
#                 dst = cv2.perspectiveTransform(pts, M)
#
#                 # Vẽ hình hộp lên ảnh kết quả MÀU
#                 img_result = cv2.polylines(img_result, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
#                 print("Đã tìm thấy và khoanh vùng đối tượng!")
#             else:
#                 print("Không thể tìm thấy Homography.")
#         else:
#             print(f"Không đủ diêm khớp tốt (chỉ có {len(good_matches)}), không thể định vị.")
#
#         # --- 7. Vẽ các diêm khớp (Sử dụng ảnh MÀU đã tải lúc đầu) ---
#         img_matches = cv2.drawMatches(
#             img_query_color, kp_query,
#             img_scene_color, kp_scene,  # Sử dụng img_scene_color
#             good_matches, None,
#             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
#         )
#
#         # --- 8. Hiển thị kết quả ---
#         plt.figure(figsize=(20, 10))
#
#         plt.subplot(1, 2, 1)
#         plt.title('Các diêm khớp tốt (Good Matches)')
#         plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
#
#         plt.subplot(1, 2, 2)
#         plt.title('Kết quả định vị biển báo')
#         plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
#
#         plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Tải ảnh (Chỉ tải 1 LẦN DUY NHẤT) ---
img_query_color = cv2.imread('./img/qc-2.jpg', cv2.IMREAD_COLOR)
img_scene_color = cv2.imread('./img/img_6.png', cv2.IMREAD_COLOR)

if img_query_color is None or img_scene_color is None:
    print("Lỗi: Không thể tải ảnh. Hãy kiểm tra lại đường dẫn file.")
else:
    # Chuyển sang ảnh XÁM để xử lý
    img_query = cv2.cvtColor(img_query_color, cv2.COLOR_BGR2GRAY)
    img_scene = cv2.cvtColor(img_scene_color, cv2.COLOR_BGR2GRAY)

    # --- 2. Khởi tạo SIFT Detector ---
    # *** ĐÂY LÀ THAY ĐỔI SO VỚI ORB ***
    sift = cv2.SIFT_create()

    # --- 3. Tìm Keypoints và Descriptors (trên ảnh xám) ---
    # Cú pháp vẫn y hệt
    kp_query, des_query = sift.detectAndCompute(img_query, None)
    kp_scene, des_scene = sift.detectAndCompute(img_scene, None)

    if des_query is None or des_scene is None:
        print("Không tìm thấy đủ đặc trưng (descriptors) trong một trong hai ảnh.")
    else:
        # --- 4. Khởi tạo Brute-Force Matcher ---
        # *** ĐÂY LÀ THAY ĐỔI QUAN TRỌNG ***
        # Dùng cv2.NORM_L2 cho SIFT/SURF (vector số thực)
        # Dùng cv2.NORM_HAMMING cho ORB/BRIEF (vector nhị phân)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        matches = bf.knnMatch(des_query, des_scene, k=2)

        # --- 5. Lọc diêm khớp (Ratio Test) ---
        good_matches = []
        for item in matches:
            if len(item) == 2:
                m, n = item
                # Tỷ lệ 0.75 là tỷ lệ chuẩn của Lowe (tác giả SIFT)
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        print(f"Tìm thấy {len(good_matches)} diêm khớp tốt (SIFT).")

        # Tạo một ảnh kết quả (ảnh màu) để vẽ lên
        img_result = img_scene_color.copy()

        # --- 6. Vẽ hình hộp định vị ---
        MIN_MATCH_COUNT = 7

        if len(good_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                h, w = img_query.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                img_result = cv2.polylines(img_result, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                print("Đã tìm thấy và khoanh vùng đối tượng!")
            else:
                print("Không thể tìm thấy Homography.")
        else:
            print(f"Không đủ diêm khớp tốt (chỉ có {len(good_matches)}), không thể định vị.")

        # --- 7. Vẽ các diêm khớp ---
        img_matches = cv2.drawMatches(
            img_query_color, kp_query,
            img_scene_color, kp_scene,
            good_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # --- 8. Hiển thị kết quả ---
        plt.figure(figsize=(20, 10))

        plt.subplot(1, 2, 1)
        plt.title('Các diêm khớp tốt (SIFT)')
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))

        plt.subplot(1, 2, 2)
        plt.title('Kết quả định vị biển báo (SIFT)')
        plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))

        plt.show()