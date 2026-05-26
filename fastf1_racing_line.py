"""
FastF1 Racing Line Extractor
=============================
Pulls real GPS telemetry data for Verstappen at Monza
and visualizes the racing line colored by speed.

HOW TO USE:
  Run: python fastf1_racing_line.py
  Then open: output/racing_line_interactive.html
"""

from pathlib import Path
import numpy as np
import fastf1
import fastf1.plotting
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import os

BASE_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR  = BASE_DIR / "data" / "fastf1_cache"

# CONFIG
YEAR    = 2023
GP      = "Monza"
SESSION = "Q"        # Q = Qualifying (fastest single lap)
DRIVER  = "VER"


def setup():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))


def load_lap():
    print(f"Loading {YEAR} {GP} {SESSION} — {DRIVER}")
    print("(First run downloads ~50MB of data, cached after that)\n")

    session = fastf1.get_session(YEAR, GP, SESSION)
    session.load()

    driver_laps = session.laps.pick_drivers(DRIVER)
    fastest_lap = driver_laps.pick_fastest()

    print(f"Fastest lap: {fastest_lap['LapTime']}")
    print(f"Lap number:  {fastest_lap['LapNumber']}\n")

    return fastest_lap


def get_telemetry(lap):
    """Get full telemetry with position + speed data."""
    tel = lap.get_telemetry()

    # FastF1 gives us:
    # X, Y     — GPS coordinates (meters)
    # Speed    — km/h
    # Throttle — 0-100%
    # Brake    — boolean
    # nGear    — gear number

    print(f"Telemetry points: {len(tel)}")
    print(f"Speed range: {tel['Speed'].min():.0f} — {tel['Speed'].max():.0f} km/h")
    print(f"Columns available: {list(tel.columns)}\n")

    return tel


def plot_static(tel):
    """
    Create a stunning static matplotlib visualization.
    Line colored by speed — red=slow, green=fast.
    """
    x     = tel["X"].values
    y     = tel["Y"].values
    speed = tel["Speed"].values

    # Normalize speed to 0-1 for coloring
    speed_norm = (speed - speed.min()) / (speed.max() - speed.min())

    # Create line segments
    points   = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#0D0D0D")
    ax.set_facecolor("#0D0D0D")

    # Color map: red (slow) → yellow (medium) → green (fast)
    cmap = LinearSegmentedColormap.from_list(
        "speed", ["#E8002D", "#FFC906", "#00D2BE"]
    )

    lc = LineCollection(segments, cmap=cmap, linewidth=3, alpha=0.95)
    lc.set_array(speed_norm[:-1])
    ax.add_collection(lc)

    # Colorbar
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=speed.min(), vmax=speed.max())
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Speed (km/h)", color="white", fontsize=12)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlim(x.min() - 100, x.max() + 100)
    ax.set_ylim(y.min() - 100, y.max() + 100)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title(
        f"Verstappen — Monza {YEAR} Qualifying\nRacing Line colored by Speed",
        color="white", fontsize=16, fontweight="bold", pad=20
    )

    # Add speed legend annotations
    ax.annotate("Slow", xy=(0.02, 0.05), xycoords="axes fraction",
                color="#E8002D", fontsize=11, fontweight="bold")
    ax.annotate("Fast", xy=(0.12, 0.05), xycoords="axes fraction",
                color="#00D2BE", fontsize=11, fontweight="bold")

    out_path = OUTPUT_DIR / "racing_line_static.png"
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Static plot saved: {out_path}")


def plot_interactive(tel):
    """
    Create an interactive Plotly visualization.
    Hover over any point to see speed, throttle, gear.
    """
    x        = tel["X"].values
    y        = tel["Y"].values
    speed    = tel["Speed"].values
    throttle = tel["Throttle"].values
    gear     = tel["nGear"].values

    # Normalize speed for color
    speed_norm = (speed - speed.min()) / (speed.max() - speed.min())

    fig = go.Figure()

    # Draw line segments colored by speed
    n = len(x)
    for i in range(0, n - 1, 3):
        t = speed_norm[i]
        # Red -> Yellow -> Green based on speed
        if t < 0.5:
            r = 232
            g = int(t * 2 * 201 + 0)
            b = int(t * 2 * 45)
        else:
            r = int((1 - (t - 0.5) * 2) * 232)
            g = int(200 + (t - 0.5) * 2 * 10)
            b = int(45 + (t - 0.5) * 2 * 145)

        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))

        fig.add_trace(go.Scatter(
            x=x[i:i+4],
            y=y[i:i+4],
            mode="lines",
            line=dict(color=f"rgb({r},{g},{b})", width=3),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Invisible scatter for hover tooltips
    fig.add_trace(go.Scatter(
        x=x[::3],
        y=y[::3],
        mode="markers",
        marker=dict(size=4, opacity=0),
        name="Telemetry",
        hovertemplate=(
            "<b>Speed:</b> %{customdata[0]:.0f} km/h<br>"
            "<b>Throttle:</b> %{customdata[1]:.0f}%<br>"
            "<b>Gear:</b> %{customdata[2]}<br>"
            "<extra></extra>"
        ),
        customdata=np.stack([
            speed[::3],
            throttle[::3],
            gear[::3],
        ], axis=-1),
    ))

    fig.update_layout(
        title=dict(
            text=f"Verstappen — Monza {YEAR} Qualifying | Racing Line by Speed",
            font=dict(size=18, color="white"),
            x=0.5,
        ),
        plot_bgcolor="#0D0D0D",
        paper_bgcolor="#0D0D0D",
        font=dict(color="white"),
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        hovermode="closest",
        width=1000,
        height=750,
        annotations=[
            dict(text="🟥 Slow  🟨 Medium  🟩 Fast",
                 xref="paper", yref="paper",
                 x=0.5, y=-0.02, showarrow=False,
                 font=dict(size=13, color="white")),
        ]
    )

    out_path = OUTPUT_DIR / "racing_line_interactive.html"
    fig.write_html(str(out_path))
    print(f"Interactive plot saved: {out_path}")
    print(f"\nOpen in browser: {out_path}")


def main():
    setup()
    lap = load_lap()
    tel = get_telemetry(lap)
    print("Creating static plot...")
    plot_static(tel)
    print("Creating interactive plot...")
    plot_interactive(tel)
    print("\nAll done! Open output/racing_line_interactive.html in your browser.")


if __name__ == "__main__":
    main()
