"""
STEP 3 — Segment frames to find the track surface
==================================================
This is the core ML step.

WHAT IS SEGMENTATION?
  Normal image classification asks "what is in this image?" (e.g. "a racetrack")
  Semantic segmentation asks "what is EVERY PIXEL?" — it labels each pixel
  with a category like: road, car, sky, barrier, grass, etc.

WHAT MODEL ARE WE USING?
  DeepLabV3 with a ResNet-50 backbone, pretrained on the COCO dataset.
  It already knows what "road" looks like — we don't need to train it.
  This is called "transfer learning" — using a model trained by someone
  else on millions of images, for free.

  COCO class 0 = background
  COCO class 1 = person  ... 
  COCO class 21 = road  ← this is what we care about

HOW TO USE:
  Run:  python 3_segment_frames.py
"""

import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from PIL import Image
import numpy as np
import os
import json

# ── CONFIG ──────────────────────────────────────────────────────────────────
INPUT_FOLDER  = "data/raw_frames"
OUTPUT_FOLDER = "data/processed"

# COCO dataset class index for "road"
ROAD_CLASS_INDEX = 21

# Only process every Nth frame to save time during development.
# Set to 1 to process all frames once you're happy with results.
PROCESS_EVERY_N = 3
# ────────────────────────────────────────────────────────────────────────────


def load_model():
    """
    Load the pretrained DeepLabV3 segmentation model.

    The first time you run this it will download ~160MB of model weights.
    After that it's cached and loads instantly.
    """
    print("Loading segmentation model (downloads ~160MB on first run)...")

    # Load pretrained weights
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model   = deeplabv3_resnet50(weights=weights)

    # Set to evaluation mode — this disables dropout/batchnorm training behavior
    model.eval()

    # Use GPU if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    model = model.to(device)

    return model, device, weights.transforms()


def segment_frame(model, device, preprocess, image_path: str) -> np.ndarray:
    """
    Run the segmentation model on a single frame.
    Returns a binary mask where True = road/track pixels.

    A "mask" is just a 2D array the same size as the image,
    where each cell is True (track) or False (not track).
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    
    # Preprocess: resize, normalize pixel values to what the model expects
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    # unsqueeze(0) adds a batch dimension — models expect [batch, channels, H, W]

    # Run the model — torch.no_grad() tells PyTorch we don't need gradients
    # (gradients are only needed during training, not inference)
    with torch.no_grad():
        output = model(input_tensor)["out"]  # shape: [1, 21, H, W]

    # output has 21 channels, one per class. Take the argmax to get the
    # predicted class for each pixel.
    predicted_classes = output.argmax(dim=1).squeeze(0).cpu().numpy()  # shape: [H, W]

    # Create binary mask: True where the model predicted "road"
    road_mask = (predicted_classes == ROAD_CLASS_INDEX)

    return road_mask.astype(np.uint8) * 255  # Convert to 0/255 for saving as image


def find_car_bottom_center(frame_path: str) -> tuple[int, int] | None:
    """
    Find the approximate bottom-center of the car in the frame.

    In onboard F1 footage, the car's nose/chassis is visible at the 
    bottom-center of the frame. We use this as the wheel contact point.

    Simple approach: look at the bottom 30% of the frame, find the 
    horizontal center of the darkest region (the car body).
    """
    img = np.array(Image.open(frame_path).convert("L"))  # grayscale
    h, w = img.shape

    # Focus on the bottom 30% of the frame where the car is
    bottom_region = img[int(h * 0.7):, :]

    # Find pixels darker than threshold (car body is usually dark)
    dark_mask = bottom_region < 60

    if dark_mask.sum() < 100:  # not enough dark pixels found
        return None

    # Find center of mass of dark pixels
    rows, cols = np.where(dark_mask)
    center_x = int(np.mean(cols))
    center_y = int(h * 0.7 + np.mean(rows))

    return (center_x, center_y)


def process_all_frames(model, device, preprocess) -> list[dict]:
    """
    Process all extracted frames and collect:
    - The road mask for each frame
    - The estimated car position in each frame
    """
    frame_files = sorted([
        f for f in os.listdir(INPUT_FOLDER) 
        if f.endswith(".jpg")
    ])

    if not frame_files:
        print(f"No frames found in {INPUT_FOLDER}/")
        print("Run 2_extract_frames.py first.")
        return []

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print(f"Found {len(frame_files)} frames. Processing every {PROCESS_EVERY_N}th frame...")
    print("(This is the slow step — a few seconds per frame on CPU)\n")

    results = []
    frames_to_process = frame_files[::PROCESS_EVERY_N]

    for i, filename in enumerate(frames_to_process):
        frame_path = os.path.join(INPUT_FOLDER, filename)

        # Get road segmentation mask
        mask = segment_frame(model, device, preprocess, frame_path)

        # Save the mask as an image so you can inspect it
        mask_filename = filename.replace(".jpg", "_mask.png")
        mask_path = os.path.join(OUTPUT_FOLDER, mask_filename)
        Image.fromarray(mask).save(mask_path)

        # Get car position estimate
        car_pos = find_car_bottom_center(frame_path)

        results.append({
            "frame":    filename,
            "mask":     mask_filename,
            "car_pos":  car_pos,
        })

        # Progress update
        if (i + 1) % 20 == 0 or (i + 1) == len(frames_to_process):
            pct = ((i + 1) / len(frames_to_process)) * 100
            print(f"  {pct:.0f}% — processed {i+1}/{len(frames_to_process)} frames")

    # Save results to JSON so next steps can load them
    results_path = os.path.join(OUTPUT_FOLDER, "frame_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone! Results saved to: {results_path}")
    print(f"Next step: run  python 4_extract_line.py")
    return results


if __name__ == "__main__":
    model, device, preprocess = load_model()
    process_all_frames(model, device, preprocess)
