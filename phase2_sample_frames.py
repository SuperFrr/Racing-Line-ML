"""
PHASE 2 — Sample frames for Roboflow labeling
==============================================
Picks 150 evenly spaced frames from your raw_frames folder
and copies them to a new folder ready to upload to Roboflow.

We don't label all 580 — that would take forever.
150 is the sweet spot for a good fine-tuned model.

HOW TO USE:
  Run:  python phase2_sample_frames.py
  Then upload everything in data/label_these/ to Roboflow
"""

import os
import shutil

# ── CONFIG ──────────────────────────────────────────────────────────────────
INPUT_FOLDER  = "data/raw_frames"
OUTPUT_FOLDER = "data/label_these"
NUM_FRAMES    = 150
# ────────────────────────────────────────────────────────────────────────────


def sample_frames(input_folder: str, output_folder: str, num_frames: int) -> None:
    all_frames = sorted([
        f for f in os.listdir(input_folder)
        if f.endswith(".jpg")
    ])

    if len(all_frames) == 0:
        print("No frames found. Run 2_extract_frames.py first.")
        return

    # Pick evenly spaced frames across the whole video
    step = max(1, len(all_frames) // num_frames)
    selected = all_frames[::step][:num_frames]

    os.makedirs(output_folder, exist_ok=True)

    print(f"Selecting {len(selected)} frames from {len(all_frames)} total...")

    for i, fname in enumerate(selected):
        src = os.path.join(input_folder, fname)
        dst = os.path.join(output_folder, fname)
        shutil.copy2(src, dst)

        if (i + 1) % 25 == 0:
            print(f"  Copied {i+1}/{len(selected)} frames")

    print(f"\nDone! {len(selected)} frames saved to: {output_folder}/")
    print("Next: upload this folder to Roboflow for labeling")


if __name__ == "__main__":
    sample_frames(INPUT_FOLDER, OUTPUT_FOLDER, NUM_FRAMES)
