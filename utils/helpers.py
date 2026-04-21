"""
utils/helpers.py
================
Shared helper functions used across the pipeline scripts.
"""

import os
import json
import numpy as np
from PIL import Image


def load_json(path: str) -> dict | list:
    """Load a JSON file and return its contents."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: dict | list, path: str) -> None:
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_image_as_array(path: str, grayscale: bool = False) -> np.ndarray:
    """Load an image file as a numpy array."""
    mode = "L" if grayscale else "RGB"
    return np.array(Image.open(path).convert(mode))


def check_step_complete(path: str, step_name: str) -> bool:
    """
    Check if an expected output file exists.
    Prints a helpful message if it doesn't.
    """
    if os.path.exists(path):
        return True
    print(f"Missing file: {path}")
    print(f"Complete '{step_name}' before running this step.")
    return False


def normalize_positions(positions: list[tuple], 
                        target_width: int = 800, 
                        target_height: int = 600) -> list[tuple]:
    """
    Normalize a list of (x, y) positions to fit within target dimensions.
    Useful when combining data from videos of different resolutions.
    """
    if not positions:
        return []

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_range = x_max - x_min or 1
    y_range = y_max - y_min or 1

    normalized = [
        (
            int((x - x_min) / x_range * target_width),
            int((y - y_min) / y_range * target_height),
        )
        for x, y in positions
    ]
    return normalized
