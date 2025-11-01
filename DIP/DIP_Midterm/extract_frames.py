"""
Extract 3 output frames from task1_output.mp4 with at least 8 seconds between them.
These frames will show the detected traffic signs with bounding boxes.
"""

import cv2
import os

def extract_frames_for_submission(video_path, output_dir="output_frames", min_gap_seconds=8):
    """
    Extract 3 frames from the output video with minimum gap between them.
    
    Args:
        video_path: Path to the output video (task1_output.mp4)
        output_dir: Directory to save extracted frames
        min_gap_seconds: Minimum time gap between frames (default: 8 seconds)
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
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Min gap between frames: {min_gap_seconds} seconds")
    print()
    
    # Calculate frame indices
    # We want 3 frames spread across the video with at least min_gap_seconds between them
    min_gap_frames = int(min_gap_seconds * fps)
    
    # Strategy: Extract frames at beginning, middle, and end with proper spacing
    # Frame 1: ~10 seconds in (to ensure signs are visible)
    # Frame 2: Middle of video
    # Frame 3: ~10 seconds before end
    
    frame1_time = 10.0  # 10 seconds
    frame2_time = duration / 2  # Middle
    frame3_time = duration - 10.0  # 10 seconds before end
    
    # Ensure minimum gap
    if frame2_time - frame1_time < min_gap_seconds:
        frame2_time = frame1_time + min_gap_seconds
    
    if frame3_time - frame2_time < min_gap_seconds:
        frame3_time = frame2_time + min_gap_seconds
    
    # Make sure frame3 doesn't exceed video duration
    if frame3_time > duration - 1:
        frame3_time = duration - 1
    
    frame_times = [frame1_time, frame2_time, frame3_time]
    frame_indices = [int(t * fps) for t in frame_times]
    
    print("Extracting frames at:")
    for i, (idx, t) in enumerate(zip(frame_indices, frame_times), 1):
        print(f"  Frame {i}: Time {t:.2f}s (Frame #{idx})")
    print()
    
    # Extract frames
    extracted = 0
    for i, (frame_idx, frame_time) in enumerate(zip(frame_indices, frame_times), 1):
        # Set position to desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read frame
        ret, frame = cap.read()
        
        if ret:
            # Save frame
            output_path = os.path.join(output_dir, f"output_frame_{i}_t{frame_time:.1f}s.png")
            cv2.imwrite(output_path, frame)
            print(f"✓ Saved: {output_path}")
            extracted += 1
        else:
            print(f"✗ Error: Could not read frame at index {frame_idx}")
    
    cap.release()
    
    print(f"\nSuccessfully extracted {extracted}/3 frames")
    print(f"Frames saved in: {output_dir}/")
    
    # Print gap verification
    print("\nGap Verification:")
    for i in range(len(frame_times) - 1):
        gap = frame_times[i+1] - frame_times[i]
        print(f"  Gap between Frame {i+1} and Frame {i+2}: {gap:.2f} seconds ✓" if gap >= min_gap_seconds else f"  Gap: {gap:.2f} seconds ✗")


if __name__ == "__main__":
    # Input video (your processed output with bounding boxes)
    video_path = "task1_output.mp4"
    
    # Output directory for frames
    output_dir = "output_frames"
    
    # Minimum gap between frames (8 seconds as per requirement)
    min_gap = 8
    
    print("="*60)
    print("Task 1: Extract Output Frames for Submission")
    print("="*60)
    print()
    
    extract_frames_for_submission(video_path, output_dir, min_gap)
    
    print("\n" + "="*60)
    print("Done! Check the 'output_frames' folder for your 3 frames.")
    print("="*60)
