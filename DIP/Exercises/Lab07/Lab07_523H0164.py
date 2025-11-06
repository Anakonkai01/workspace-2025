import cv2
import numpy as np
import os
from datetime import datetime

STUDENT_ID = "523H0164"

# Global variables for mouse drawing
drawing = False
ix, iy = -1, -1

def add_student_id(frame):
    h, w = frame.shape[:2]
    cv2.putText(frame, STUDENT_ID, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

# 1. Play a video using OpenCV
def play_video(video_path):
    print("Playing video")
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read(
        if not ret:
            break
        frame = add_student_id(frame)
        cv2.imshow('Play Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# 2. Extract images from video
def extract_images(video_path):
    print("Extracting images")
    if not os.path.exists('frames'):
        os.makedirs('frames')
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % 30 == 0:
            cv2.imwrite(f'frames/frame_{saved:04d}.jpg', frame)
            saved += 1
        count += 1
    cap.release()
    print(f"Extracted {saved} frames to 'frames' folder")

# 3. Create video using multiple images
def create_video_from_images():
    print("Creating video from images")
    if not os.path.exists('frames') or len(os.listdir('frames')) == 0:
        print("No frames found. Skip this task.")
        return
    
    images = sorted([img for img in os.listdir('frames') if img.endswith('.jpg')])
    if not images:
        print("No images found")
        return
    
    frame = cv2.imread(os.path.join('frames', images[0]))
    h, w, _ = frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_from_images.mp4', fourcc, 20.0, (w, h))
    
    for image in images:
        img = cv2.imread(os.path.join('frames', image))
        out.write(img)
    
    out.release()
    print("Video created: output_from_images.mp4")

# 4. Capture Video from Camera
def capture_from_camera():
    print("Capturing from camera (press 'q' to stop)")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot access camera")
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('camera_output.mp4', fourcc, 20.0, (640, 480))
    
    frame_count = 0
    while frame_count < 100:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = add_student_id(frame)
        out.write(frame)
        cv2.imshow('Camera', frame)
        
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Camera video saved: camera_output.mp4")

# 5. Process images of a video
def process_video_images(video_path):
    print("Processing video images")
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed = cv2.GaussianBlur(frame, (15, 15), 0)
        processed = add_student_id(processed)
        cv2.imshow('Processed Video', processed)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 6. Writing to video with OpenCV
def write_to_video(video_path):
    print("Writing to video file")
    cap = cv2.VideoCapture(video_path)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output_written.mp4', fourcc, 20.0, (w, h))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = add_student_id(frame)
        out.write(frame)
    
    cap.release()
    out.release()
    print("Video written: output_written.mp4")

# 7. Write text on video
def write_text_on_video(video_path):
    print("Writing text on video")
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.putText(frame, 'Lab 07 - DIP', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        frame = add_student_id(frame)
        cv2.imshow('Text on Video', frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 8. Play video in reverse mode
def reverse_video(video_path):
    print("Playing video in reverse")
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    for frame in reversed(frames):
        frame = add_student_id(frame)
        cv2.imshow('Reverse Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

# 9. Converting color video to grayscale
def color_to_grayscale(video_path):
    print("Converting to grayscale")
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        gray_bgr = add_student_id(gray_bgr)
        cv2.imshow('Grayscale', gray_bgr)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 10. Displaying real time FPS
def display_real_time_fps(video_path):
    print("Displaying real-time FPS")
    cap = cv2.VideoCapture(video_path)
    prev_time = datetime.now()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        curr_time = datetime.now()
        time_diff = (curr_time - prev_time).total_seconds()
        fps = 1 / time_diff if time_diff > 0 else 0
        prev_time = curr_time
        
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        frame = add_student_id(frame)
        cv2.imshow('Real-time FPS', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 11. Get video duration
def get_video_duration(video_path):
    print("Getting video duration")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"FPS: {fps}")
    print(f"Total Frames: {frame_count}")
    print(f"Duration: {duration:.2f} seconds")
    
    cap.release()

# 12. Click response on video using Events
def mouse_click_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({x}, {y})")

def click_response_video(video_path):
    print("Click on video (left click to see coordinates)")
    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow('Click on Video')
    cv2.setMouseCallback('Click on Video', mouse_click_callback)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = add_student_id(frame)
        cv2.imshow('Click on Video', frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 13. Creating slow motion video
def slow_motion_video(video_path):
    print("Playing slow motion")
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = add_student_id(frame)
        cv2.imshow('Slow Motion', frame)
        
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 14. Save frames of live video with timestamps
def save_frames_with_timestamps():
    print("Saving frames with timestamps (press 'q' to stop)")
    if not os.path.exists('timestamp_frames'):
        os.makedirs('timestamp_frames')
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access camera")
        return
    
    frame_count = 0
    while frame_count < 50:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f'timestamp_frames/frame_{timestamp}.jpg'
        cv2.imwrite(filename, frame)
        
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        frame = add_student_id(frame)
        cv2.imshow('Saving Frames', frame)
        
        frame_count += 1
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {frame_count} frames with timestamps")

# 15. Change video resolution
def change_video_resolution(video_path):
    print("Changing video resolution")
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        resized = cv2.resize(frame, (640, 480))
        resized = add_student_id(resized)
        cv2.imshow('Changed Resolution (640x480)', resized)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 16. Faster video file FPS
def faster_video_fps(video_path):
    print("Playing faster (higher FPS)")
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = add_student_id(frame)
        cv2.imshow('Faster Video', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 17. Saving key event video clips
def save_key_event_clips(video_path):
    print("Press 's' to save clip, 'q' to quit")
    cap = cv2.VideoCapture(video_path)
    clip_number = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = add_student_id(frame)
        cv2.imshow('Key Event Clips', frame)
        
        key = cv2.waitKey(25) & 0xFF
        if key == ord('s'):
            filename = f'clip_{clip_number:03d}.jpg'
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            clip_number += 1
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 18. Detect shapes in images
def detect_shapes(video_path):
    print("Detecting shapes")
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)
            
            if len(approx) == 3:
                shape = "Triangle"
            elif len(approx) == 4:
                shape = "Rectangle"
            elif len(approx) > 4:
                shape = "Circle"
            else:
                shape = "Shape"
            
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(frame, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        frame = add_student_id(frame)
        cv2.imshow('Shape Detection', frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 19. Denoising of colored images
def denoise_colored_images(video_path):
    print("Denoising colored images")
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        denoised = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        denoised = add_student_id(denoised)
        cv2.imshow('Denoised', denoised)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 20. Drawing with mouse on images
def draw_with_mouse(event, x, y, flags, param):
    global drawing, ix, iy
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(param, (ix, iy), (x, y), (0, 255, 0), 3)
            ix, iy = x, y
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def drawing_with_mouse(video_path):
    global drawing
    print("Drawing with mouse (drag to draw, 'q' to quit)")
    cap = cv2.VideoCapture(video_path)
    
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return
    
    canvas = frame.copy()
    cv2.namedWindow('Draw on Image')
    cv2.setMouseCallback('Draw on Image', draw_with_mouse, canvas)
    
    while True:
        cv2.imshow('Draw on Image', canvas)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            ret, canvas = cap.read()
            if not ret:
                break
    
    cap.release()
    cv2.destroyAllWindows()

# 21. Measure similarity between images
def measure_similarity(video_path):
    print("Measuring similarity between frames")
    cap = cv2.VideoCapture(video_path)
    
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        mse = np.mean((prev_gray.astype(float) - gray.astype(float)) ** 2)
        similarity = 1 - (mse / (255 ** 2))
        
        cv2.putText(frame, f'Similarity: {similarity:.4f}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        frame = add_student_id(frame)
        cv2.imshow('Frame Similarity', frame)
        
        prev_gray = gray
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 22. Display multiple images in one window
def display_multiple_images(video_path):
    print("Displaying multiple images in one window")
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        edges = cv2.Canny(gray, 50, 150)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        blur = cv2.GaussianBlur(frame, (15, 15), 0)
        
        h, w = 240, 320
        img1 = cv2.resize(frame, (w, h))
        img2 = cv2.resize(gray_bgr, (w, h))
        img3 = cv2.resize(edges_bgr, (w, h))
        img4 = cv2.resize(blur, (w, h))
        
        top_row = np.hstack([img1, img2])
        bottom_row = np.hstack([img3, img4])
        combined = np.vstack([top_row, bottom_row])
        
        combined = add_student_id(combined)
        cv2.imshow('Multiple Images', combined)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 23. Concatenate images
def concatenate_images(video_path):
    print("Concatenating images")
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        concatenated = np.hstack([frame, gray_bgr])
        concatenated = add_student_id(concatenated)
        cv2.imshow('Concatenated', concatenated)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 24. Adding borders to images
def add_borders(video_path):
    print("Adding borders to images")
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        bordered = cv2.copyMakeBorder(frame, 20, 20, 20, 20, 
                                     cv2.BORDER_CONSTANT, value=[0, 255, 0])
        bordered = add_student_id(bordered)
        cv2.imshow('Bordered', bordered)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 25. Creating hybrid images
def create_hybrid_images(video_path):
    print("Creating hybrid images")
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        low_pass = cv2.GaussianBlur(frame, (21, 21), 0)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        high_pass = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        hybrid = cv2.addWeighted(low_pass, 0.7, high_pass, 0.3, 0)
        hybrid = add_student_id(hybrid)
        cv2.imshow('Hybrid Image', hybrid)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# Main program
if __name__ == "__main__":
    print("Lab 07")
    print("Student:", STUDENT_ID)
    print()
    
    video_path = 'video.mp4'
    
    if not os.path.exists(video_path):
        print("Error: video.mp4 not found")
        exit(1)
    
    print("Video:", video_path)
    print("Press 'q' to skip")
    print()
    
    print("Task 1: Play video")
    play_video(video_path)
    
    print("Task 2: Extract images")
    extract_images(video_path)
    
    print("Task 3: Create video from images")
    create_video_from_images()
    
    print("Task 4: Capture from camera")
    capture_from_camera()
    
    print("Task 5: Process video")
    process_video_images(video_path)
    
    print("Task 6: Write to video")
    write_to_video(video_path)
    
    print("Task 7: Write text")
    write_text_on_video(video_path)
    
    print("Task 8: Reverse video")
    reverse_video(video_path)
    
    print("Task 9: Grayscale")
    color_to_grayscale(video_path)
    
    print("Task 10: Real-time FPS")
    display_real_time_fps(video_path)
    
    print("Task 11: Video duration")
    get_video_duration(video_path)
    
    print("Task 12: Click response")
    click_response_video(video_path)
    
    print("Task 13: Slow motion")
    slow_motion_video(video_path)
    
    print("Task 14: Save with timestamps")
    save_frames_with_timestamps()
    
    print("Task 15: Change resolution")
    change_video_resolution(video_path)
    
    print("Task 16: Faster FPS")
    faster_video_fps(video_path)
    
    print("Task 17: Key event clips")
    save_key_event_clips(video_path)
    
    print("Task 18: Detect shapes")
    detect_shapes(video_path)
    
    print("Task 19: Denoise")
    denoise_colored_images(video_path)
    
    print("Task 20: Draw with mouse")
    drawing_with_mouse(video_path)
    
    print("Task 21: Measure similarity")
    measure_similarity(video_path)
    
    print("Task 22: Multiple images")
    display_multiple_images(video_path)
    
    print("Task 23: Concatenate")
    concatenate_images(video_path)
    
    print("Task 24: Add borders")
    add_borders(video_path)
    
    print("Task 25: Hybrid images")
    create_hybrid_images(video_path)
