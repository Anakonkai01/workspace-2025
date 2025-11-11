import cv2
import os

DES_DIR = 'resized_images'
if not os.path.exists(DES_DIR):
    os.makedirs(DES_DIR)
TARGET_SIZE = (100, 100)


for file_name in os.listdir('.'):
    
    if file_name.endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.imread(file_name)
        resized_img = cv2.resize(img, TARGET_SIZE)
        cv2.imwrite(os.path.join(DES_DIR, file_name), resized_img)