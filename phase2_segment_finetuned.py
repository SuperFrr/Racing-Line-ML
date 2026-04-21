"""
PHASE 2 — Updated 3_segment_frames.py
======================================
Drop-in replacement for 3_segment_frames.py that loads YOUR
fine-tuned model instead of the generic pretrained one.

HOW TO USE:
  After training completes and models/finetuned_model.pth exists,
  replace your 3_segment_frames.py with this file and re-run
  the full pipeline from step 3.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from PIL import Image
import numpy as np
import os
import json

# ── CONFIG ──────────────────────────────────────────────────────────────────
INPUT_FOLDER    = "data/raw_frames"
OUTPUT_FOLDER   = "data/processed"
MODEL_PATH      = "models/finetuned_model.pth"
NUM_CLASSES     = 3       # background, track_surface, kerb
TRACK_CLASS_IDX = 1       # class index for track surface
KERB_CLASS_IDX  = 2       # class index for kerb
PROCESS_EVERY_N = 3
# ────────────────────────────────────────────────────────────────────────────


def load_finetuned_model(model_path: str, num_classes: int):
    """Load your fine-tuned model weights."""
    print(f"Loading fine-tuned model from: {model_path}")

    # Build same architecture as training
    model = deeplabv3_resnet50(weights=None)
    model.classifier[4]     = nn.Conv2d(256, num_classes, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    # Load your trained weights
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    print(f"Model loaded — using device: {device}")
    return model, device


def get_transforms():
    return T.Compose([
        T.Resize((520, 520)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def segment_frame(model, device, transforms, image_path: str):
    """Run segmentation and return track + kerb masks."""
    img          = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    input_tensor = transforms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)["out"]

    predicted = output.argmax(dim=1).squeeze(0).cpu().numpy()

    # Resize mask back to original image size
    mask_img   = Image.fromarray(predicted.astype(np.uint8))
    mask_img   = mask_img.resize((orig_w, orig_h), Image.NEAREST)
    mask_array = np.array(mask_img)

    # Separate masks for track and kerbs
    track_mask = (mask_array == TRACK_CLASS_IDX).astype(np.uint8) * 255
    kerb_mask  = (mask_array == KERB_CLASS_IDX).astype(np.uint8) * 255

    return track_mask, kerb_mask


def find_car_position(frame_path: str, track_mask: np.ndarray):
    """
    Find car position using the track mask.
    Look for the bottom-center of the detected track region —
    that's where the car's tyres contact the surface.
    """
    h, w = track_mask.shape

    # Focus on bottom half of frame
    bottom_mask = track_mask[h//2:, :]
    rows, cols  = np.where(bottom_mask > 0)

    if len(cols) < 50:
        return None

    # Center of the bottom track region
    center_x = int(np.median(cols))
    center_y = int(h//2 + np.median(rows))
    return (center_x, center_y)


def process_all_frames(model, device, transforms):
    frame_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith(".jpg")])

    if not frame_files:
        print(f"No frames found in {INPUT_FOLDER}/")
        return []

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Processing {len(frame_files)} frames (every {PROCESS_EVERY_N}th)...\n")

    results          = []
    frames_to_process = frame_files[::PROCESS_EVERY_N]

    for i, filename in enumerate(frames_to_process):
        frame_path = os.path.join(INPUT_FOLDER, filename)

        track_mask, kerb_mask = segment_frame(model, device, transforms, frame_path)

        # Save masks
        Image.fromarray(track_mask).save(
            os.path.join(OUTPUT_FOLDER, filename.replace(".jpg", "_track.png"))
        )
        Image.fromarray(kerb_mask).save(
            os.path.join(OUTPUT_FOLDER, filename.replace(".jpg", "_kerb.png"))
        )

        car_pos = find_car_position(frame_path, track_mask)

        results.append({
            "frame":   filename,
            "car_pos": car_pos,
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(frames_to_process):
            pct = ((i + 1) / len(frames_to_process)) * 100
            print(f"  {pct:.0f}% — {i+1}/{len(frames_to_process)} frames")

    results_path = os.path.join(OUTPUT_FOLDER, "frame_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone! Results saved to: {results_path}")
    print("Next: run  python 4_extract_line.py")
    return results


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Fine-tuned model not found at: {MODEL_PATH}")
        print("Complete phase2_finetune.py training first.")
    else:
        model, device = load_finetuned_model(MODEL_PATH, NUM_CLASSES)
        transforms    = get_transforms()
        process_all_frames(model, device, transforms)
