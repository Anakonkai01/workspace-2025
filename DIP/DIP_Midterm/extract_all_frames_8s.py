"""
Extract ALL frames at 8-second intervals from task1_output.mp4
This gives you multiple options to choose the best 3 frames for submission.
"""

import cv2
import os

def extract_frames_every_8_seconds(video_path, output_dir="all_output_frames", interval_seconds=8):
    """
    Extract frames at regular intervals from the output video.
    
    Args:
        video_path: Path to the output video (task1_output.mp4)
        output_dir: Directory to save extracted frames
        interval_seconds: Time interval between frames (default: 8 seconds)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video Properties:")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Extraction interval: Every {interval_seconds} seconds")
    print()
    
    # Calculate frame indices at 8-second intervals
    frame_interval = int(interval_seconds * fps)
    
    # Start from a few seconds in (to skip potential black frames at start)
    start_time = 2  # Start at 2 seconds
    start_frame = int(start_time * fps)
    
    frame_indices = []
    current_frame = start_frame
    current_time = start_time
    
    while current_frame < total_frames:
        frame_indices.append((current_frame, current_time))
        current_frame += frame_interval
        current_time += interval_seconds
    
    print(f"Will extract {len(frame_indices)} frames:")
    print()
    
    # Extract frames
    extracted = 0
    for i, (frame_idx, frame_time) in enumerate(frame_indices, 1):
        # Set position to desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read frame
        ret, frame = cap.read()
        
        if ret:
            # Save frame with descriptive filename
            output_path = os.path.join(output_dir, f"frame_{i:02d}_t{frame_time:05.1f}s.png")
            cv2.imwrite(output_path, frame)
            print(f"✓ Frame {i:2d}: Time {frame_time:6.1f}s (Frame #{frame_idx:4d}) → {os.path.basename(output_path)}")
            extracted += 1
        else:
            print(f"✗ Error: Could not read frame at index {frame_idx}")
    
    cap.release()
    
    print()
    print("="*70)
    print(f"Successfully extracted {extracted}/{len(frame_indices)} frames")
    print(f"Frames saved in: {output_dir}/")
    print("="*70)
    print()
    print("📝 Instructions:")
    print("  1. Open the 'all_output_frames' folder")
    print("  2. Review all frames and select the best 3 that:")
    print("     - Show clear traffic sign detections")
    print("     - Have good bounding box quality")
    print("     - Represent different parts of the video")
    print("  3. Any 3 frames you select will be at least 8 seconds apart!")
    print()


if __name__ == "__main__":
    # Input video (your processed output with bounding boxes)
    video_path = "523H0164_523H0177_523H0145.mp4"
    
    # Output directory for frames
    output_dir = "all_output_frames"
    
    # Interval (8 seconds as per requirement)
    interval = 8
    
    print("="*70)
    print("Extract ALL Frames at 8-Second Intervals")
    print("="*70)
    print()
    
    extract_frames_every_8_seconds(video_path, output_dir, interval)
