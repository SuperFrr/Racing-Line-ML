"""
STEP 2 — Extract frames from the downloaded video
==================================================
A video is just a sequence of images (frames) played fast.
We pull one frame every N frames — we don't need every single frame,
just enough to capture the car's position around the track.

HOW TO USE:
  Run:  python 2_extract_frames.py
"""

import cv2          # OpenCV — the standard library for working with images/video
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────
VIDEO_PATH    = "data/video.mp4"
OUTPUT_FOLDER = "data/raw_frames"

# Extract 1 frame every N frames.
# F1 broadcasts are ~25fps. FRAME_SKIP=5 gives us 5 frames/second.
# That's plenty of data without huge storage requirements.
FRAME_SKIP = 5

# Resize frames to this width (height auto-scales). 
# Smaller = faster processing. 640px is a good balance.
RESIZE_WIDTH = 640
# ────────────────────────────────────────────────────────────────────────────


def extract_frames(video_path: str, output_folder: str, frame_skip: int, resize_width: int) -> None:
    """
    Read through the video file and save every Nth frame as a JPEG image.
    """
    # Check the video file exists
    if not os.path.exists(video_path):
        print(f"Video not found at: {video_path}")
        print("Run 1_download_video.py first.")
        return

    os.makedirs(output_folder, exist_ok=True)

    # Open the video file with OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video file. Is it a valid .mp4?")
        return

    # Get basic video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    duration_sec = total_frames / fps

    print(f"Video info:")
    print(f"  Total frames : {total_frames}")
    print(f"  FPS          : {fps:.1f}")
    print(f"  Duration     : {duration_sec:.1f} seconds ({duration_sec/60:.1f} minutes)")
    print(f"  Extracting 1 frame every {frame_skip} frames...")
    print()

    frame_count  = 0   # how many frames we've read
    saved_count  = 0   # how many frames we've saved

    while True:
        # Read the next frame from the video
        success, frame = cap.read()

        # If read() returns False, we've reached the end of the video
        if not success:
            break

        # Only save every Nth frame
        if frame_count % frame_skip == 0:
            # Resize the frame — keeps processing fast
            height, width = frame.shape[:2]
            new_height = int(height * (resize_width / width))
            frame_resized = cv2.resize(frame, (resize_width, new_height))

            # Save as JPEG
            filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame_resized)
            saved_count += 1

            # Print progress every 100 saved frames
            if saved_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"  Progress: {progress:.0f}% — saved {saved_count} frames")

        frame_count += 1

    cap.release()
    print(f"\nDone! Extracted {saved_count} frames to: {output_folder}/")
    print(f"Next step: run  python 3_segment_frames.py")


if __name__ == "__main__":
    extract_frames(VIDEO_PATH, OUTPUT_FOLDER, FRAME_SKIP, RESIZE_WIDTH)
