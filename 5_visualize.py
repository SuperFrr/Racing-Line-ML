"""
STEP 5 — Visualize the extracted racing line
=============================================
The final step — making it look great.

We produce two outputs:
  1. A static matplotlib plot showing raw positions + smooth line
  2. An interactive Plotly HTML file you can open in a browser
     (this is what you put in your portfolio)

HOW TO USE:
  Run:  python 5_visualize.py

  Then open:  output/racing_line_interactive.html  in your browser
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────
LINE_DATA_PATH  = "data/processed/racing_line.json"
STATIC_OUTPUT   = "output/racing_line_static.png"
INTERACTIVE_OUT = "output/racing_line_interactive.html"
# ────────────────────────────────────────────────────────────────────────────


def load_line_data(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def plot_static(data: dict) -> None:
    """
    Create a clean matplotlib visualization:
    - Gray dots for raw noisy positions
    - Bright green line for the smooth extracted racing line
    - Dark background to look like a real racing display
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0D0D0D")

    raw_x  = data["raw_x"]
    raw_y  = data["raw_y"]
    sm_x   = data["smooth_x"]
    sm_y   = data["smooth_y"]

    # ── Left plot: raw positions ──────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#1A1A1A")
    ax1.scatter(raw_x, raw_y, c="#555555", s=8, alpha=0.5, label="Raw car positions")
    ax1.set_title("Raw detected car positions", color="white", fontsize=13, pad=12)
    ax1.tick_params(colors="gray")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333333")
    ax1.legend(facecolor="#1A1A1A", labelcolor="white", fontsize=9)

    # Invert Y axis — in image coordinates, Y increases downward,
    # but we want "up" on screen to mean "up on track"
    ax1.invert_yaxis()
    ax1.set_xlabel("Horizontal position (px)", color="gray", fontsize=10)
    ax1.set_ylabel("Vertical position (px)", color="gray", fontsize=10)

    # ── Right plot: smooth racing line ────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#1A1A1A")

    # Background scatter of raw points for context
    ax2.scatter(raw_x, raw_y, c="#333333", s=5, alpha=0.3, label="Raw positions")

    # Color the racing line by position along the lap (blue → green)
    points = np.array([sm_x, sm_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("racing", ["#1E90FF", "#00FF88"])
    lc = LineCollection(segments, cmap=cmap, linewidth=2.5, alpha=0.9)
    lc.set_array(np.linspace(0, 1, len(segments)))
    ax2.add_collection(lc)

    ax2.set_xlim(min(sm_x) - 20, max(sm_x) + 20)
    ax2.set_ylim(min(sm_y) - 20, max(sm_y) + 20)
    ax2.invert_yaxis()
    ax2.set_title("Extracted racing line", color="white", fontsize=13, pad=12)
    ax2.tick_params(colors="gray")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333333")
    ax2.set_xlabel("Horizontal position (px)", color="gray", fontsize=10)
    ax2.set_ylabel("Vertical position (px)", color="gray", fontsize=10)

    # Add a subtle legend for line color
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#1E90FF", linewidth=2, label="Turn-in"),
        Line2D([0], [0], color="#00FF88", linewidth=2, label="Exit"),
    ]
    ax2.legend(handles=legend_elements, facecolor="#1A1A1A", labelcolor="white", fontsize=9)

    plt.suptitle("F1 Racing Line Extraction", color="white", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    os.makedirs("output", exist_ok=True)
    plt.savefig(STATIC_OUTPUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Static plot saved: {STATIC_OUTPUT}")
    plt.close()


def plot_interactive(data: dict) -> None:
    raw_x  = data["raw_x"]
    raw_y  = data["raw_y"]
    sm_x   = data["smooth_x"]
    sm_y   = data["smooth_y"]

    fig = go.Figure()

    # Raw position scatter
    fig.add_trace(go.Scatter(
        x=raw_x,
        y=[-y for y in raw_y],
        mode="markers",
        marker=dict(color="#444444", size=4, opacity=0.4),
        name="Raw positions",
    ))

    # Smooth racing line drawn as colored segments
    n = len(sm_x)
    for i in range(0, n - 1, 5):
        progress = i / n
        r = int(30 + progress * 0)
        g = int(144 + progress * 111)
        b = int(255 - progress * 167)

        fig.add_trace(go.Scatter(
            x=sm_x[i:i+6],
            y=[-y for y in sm_y[i:i+6]],
            mode="lines",
            line=dict(color=f"rgb({r},{g},{b})", width=4),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(
        title=dict(
            text="F1 Racing Line — Extracted from Onboard Footage",
            font=dict(size=18, color="white"),
            x=0.5,
        ),
        plot_bgcolor="#1A1A1A",
        paper_bgcolor="#0D0D0D",
        font=dict(color="white"),
        xaxis=dict(title="Horizontal (px)", gridcolor="#2A2A2A", zeroline=False),
        yaxis=dict(title="Track position", gridcolor="#2A2A2A", zeroline=False,
                   scaleanchor="x", scaleratio=1),
        hovermode="closest",
        width=900,
        height=600,
    )

    fig.write_html(INTERACTIVE_OUT)
    print(f"Interactive plot saved: {INTERACTIVE_OUT}")
    print(f"\nOpen this file in your browser to see your racing line!")

def main():
    if not os.path.exists(LINE_DATA_PATH):
        print(f"Racing line data not found: {LINE_DATA_PATH}")
        print("Run 4_extract_line.py first.")
        return

    print("Loading racing line data...")
    data = load_line_data(LINE_DATA_PATH)

    print("Creating static plot...")
    plot_static(data)

    print("Creating interactive plot...")
    plot_interactive(data)

    print("\nAll done! Your racing line has been extracted.")
    print(f"  Static image:      {STATIC_OUTPUT}")
    print(f"  Interactive HTML:  {INTERACTIVE_OUT}  ← open this in your browser")


if __name__ == "__main__":
    main()
