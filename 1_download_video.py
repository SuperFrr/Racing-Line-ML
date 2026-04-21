"""
STEP 1 — Download a YouTube F1 onboard clip
"""

import yt_dlp
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────
VIDEO_URL   = "https://youtu.be/wgZdpFRpZCs?si=9mTHsml9VoXfj_rJ"
OUTPUT_PATH = "data/video.mp4"
# ────────────────────────────────────────────────────────────────────────────


def download_video(url: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Downloading: {url}")
    print("This may take a minute...\n")

    ydl_opts = {
        "format":  "best[height<=720][ext=mp4]/best[height<=720]",
        "outtmpl": output_path,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\nSuccess! Saved to: {output_path}  ({size_mb:.1f} MB)")
        print("Next step: run  python 2_extract_frames.py")
    else:
        print("\nSomething went wrong — file not found after download.")


if __name__ == "__main__":
    download_video(VIDEO_URL, OUTPUT_PATH)