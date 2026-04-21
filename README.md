# Racing Line Extractor

Extract the optimal racing line from F1 onboard YouTube footage using computer vision and semantic segmentation.

## What this project does

1. **Downloads** F1 onboard YouTube clips
2. **Extracts frames** from the video at regular intervals
3. **Segments** each frame — identifying the track surface vs everything else
4. **Tracks** where the car's wheels touch the track across many frames
5. **Fits** a smooth racing line curve through the wheel position data
6. **Visualizes** the extracted line overlaid on a top-down track diagram

## Project structure

```
racing_line_extractor/
├── data/
│   ├── raw_frames/        # Extracted video frames (images)
│   └── processed/         # Cleaned/resized frames
├── models/                # Saved model weights
├── utils/                 # Helper functions
├── notebooks/             # Jupyter notebooks for exploration
├── output/                # Final visualizations
├── 1_download_video.py    # Step 1: Download YouTube clip
├── 2_extract_frames.py    # Step 2: Pull frames from video
├── 3_segment_frames.py    # Step 3: Run segmentation model
├── 4_extract_line.py      # Step 4: Find racing line from segments
├── 5_visualize.py         # Step 5: Draw the final visualization
└── requirements.txt
```

## Setup

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline step by step
python 1_download_video.py
python 2_extract_frames.py
python 3_segment_frames.py
python 4_extract_line.py
python 5_visualize.py
```

## Recommended YouTube clips to start with

Search for these — short onboard laps work best:
- "F1 onboard lap Monza" (long straights, easy segmentation)
- "F1 onboard lap Spa Eau Rouge" (iconic corner, great for demos)
- "F1 2023 onboard Lewis Hamilton Monaco"

Best clips are 1–3 minutes, single lap, minimal camera cuts.
