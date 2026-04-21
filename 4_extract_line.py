"""
STEP 4 — Extract the racing line from car position data
========================================================
Now we have a list of (x, y) car positions from each frame.
The problem: these are in camera-view coordinates, but the car
moves through the frame over time. We need to stitch these into
a continuous path — the racing line.

WHAT THIS STEP DOES:
  1. Loads all the car positions collected in step 3
  2. Filters out bad/missing detections
  3. Fits a smooth polynomial curve through the positions
  4. This curve IS the racing line

POLYNOMIAL FITTING EXPLAINED:
  Instead of connecting every noisy dot with jagged lines,
  we fit a smooth mathematical curve that best describes
  the overall path. Think of it like drawing a smooth line
  of best fit through a scatter plot.

HOW TO USE:
  Run:  python 4_extract_line.py
"""

import json
import numpy as np
from scipy.interpolate import UnivariateSpline
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────
RESULTS_PATH = "data/processed/frame_results.json"
OUTPUT_PATH  = "data/processed/racing_line.json"

# Smoothing factor for the spline curve.
# Higher = smoother but less accurate. Lower = follows data more closely.
# Start with 500 and adjust based on visual results.
SMOOTHING = 500
# ────────────────────────────────────────────────────────────────────────────


def load_car_positions(results_path: str) -> list[tuple[int, int]]:
    """
    Load the car positions from the JSON file saved in step 3.
    Filter out frames where no car was detected (car_pos is None).
    """
    with open(results_path, "r") as f:
        results = json.load(f)

    positions = []
    for r in results:
        if r["car_pos"] is not None:
            positions.append(tuple(r["car_pos"]))

    print(f"Loaded {len(results)} frames, {len(positions)} had valid car positions")
    return positions


def smooth_racing_line(positions: list[tuple[int, int]]) -> dict:
    """
    Fit a smooth curve through the car position data.

    We separate X and Y coordinates and fit a spline through each
    as a function of "time" (frame index). This gives us a smooth
    parametric curve: (x(t), y(t)).

    A SPLINE is a piecewise polynomial curve — it passes near all your
    data points but smooths out the noise.
    """
    if len(positions) < 10:
        print("Not enough position data to fit a line (need at least 10 frames)")
        return {}

    xs = np.array([p[0] for p in positions], dtype=float)
    ys = np.array([p[1] for p in positions], dtype=float)
    t  = np.arange(len(xs), dtype=float)  # "time" parameter

    # Remove outliers — positions more than 2 std devs from mean
    # (happens when the car detection gets confused by a barrier or crowd)
    x_mean, x_std = xs.mean(), xs.std()
    y_mean, y_std = ys.mean(), ys.std()

    valid = (
        (np.abs(xs - x_mean) < 2 * x_std) &
        (np.abs(ys - y_mean) < 2 * y_std)
    )

    xs_clean = xs[valid]
    ys_clean = ys[valid]
    t_clean  = t[valid]

    print(f"After outlier removal: {valid.sum()} / {len(positions)} positions kept")

    # Fit smooth splines for x(t) and y(t)
    # s= parameter controls smoothing. Higher = smoother.
    spline_x = UnivariateSpline(t_clean, xs_clean, s=SMOOTHING, k=3)
    spline_y = UnivariateSpline(t_clean, ys_clean, s=SMOOTHING, k=3)

    # Evaluate the spline at 500 evenly spaced points
    # This gives us a smooth, dense set of (x, y) points along the racing line
    t_smooth = np.linspace(t_clean.min(), t_clean.max(), 500)
    x_smooth = spline_x(t_smooth)
    y_smooth = spline_y(t_smooth)

    return {
        "raw_x":    xs_clean.tolist(),
        "raw_y":    ys_clean.tolist(),
        "smooth_x": x_smooth.tolist(),
        "smooth_y": y_smooth.tolist(),
    }


def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"Results file not found: {RESULTS_PATH}")
        print("Run 3_segment_frames.py first.")
        return

    positions = load_car_positions(RESULTS_PATH)
    if not positions:
        print("No car positions found. Check that step 3 ran correctly.")
        return

    print("\nFitting smooth racing line curve...")
    line_data = smooth_racing_line(positions)

    if line_data:
        with open(OUTPUT_PATH, "w") as f:
            json.dump(line_data, f)
        print(f"\nRacing line saved to: {OUTPUT_PATH}")
        print(f"Smooth line has {len(line_data['smooth_x'])} points")
        print(f"Next step: run  python 5_visualize.py")


if __name__ == "__main__":
    main()
