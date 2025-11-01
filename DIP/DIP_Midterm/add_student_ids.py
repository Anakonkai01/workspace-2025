"""
Add student IDs to the top-left corner of the output video.
Format: 523H0164_523H0177_523H0145
"""

import cv2
import os

def add_student_ids_to_video(input_video, output_video, student_ids="523H0164_523H0177_523H0145"):
    """
    Add student IDs text to the top-left corner of every frame in the video.
    
    Args:
        input_video: Path to input video (task1_output.mp4)
        output_video: Path to save the new video with student IDs
        student_ids: Student IDs text to display
    """
    # Open input video
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_video}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video Properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Student IDs: {student_ids}")
    print()
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Cannot create output video {output_video}")
        cap.release()
        return
    
    # Text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    text_color = (255, 255, 255)  # White
    bg_color = (0, 0, 0)  # Black background
    padding = 10
    
    # Get text size for background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(
        student_ids, font, font_scale, font_thickness
    )
    
    print("Processing frames...")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Create background rectangle for better text visibility
        cv2.rectangle(
            frame,
            (5, 5),
            (text_width + padding * 2, text_height + baseline + padding * 2),
            bg_color,
            -1  # Filled rectangle
        )
        
        # Add student IDs text
        cv2.putText(
            frame,
            student_ids,
            (padding, text_height + padding),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA
        )
        
        # Write frame to output video
        out.write(frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    # Release resources
    cap.release()
    out.release()
    
    print()
    print("="*70)
    print(f"✓ Successfully processed {frame_count} frames")
    print(f"✓ Output saved to: {output_video}")
    print("="*70)


if __name__ == "__main__":
    # Input and output paths
    input_video = "task1_output.mp4"
    output_video = "task1_output_with_ids.mp4"
    
    # Student IDs
    student_ids = "523H0164_523H0177_523H0145"
    
    print("="*70)
    print("Add Student IDs to Output Video")
    print("="*70)
    print()
    
    # Check if input file exists
    if not os.path.exists(input_video):
        print(f"Error: Input video '{input_video}' not found!")
    else:
        add_student_ids_to_video(input_video, output_video, student_ids)
        print()
        print(f"You can now use '{output_video}' for your submission!")
