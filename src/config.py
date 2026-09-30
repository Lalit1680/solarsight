"""Central configuration.

All paths, thresholds and the class list live here so there is exactly one
place to look when something changes. Screens and pipelines import from this
module instead of hard-coding values.
"""
from __future__ import annotations

from pathlib import Path

# Project root = the directory that contains app.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Trained model weights. Gitignored: produced by training/train_yolo.py
# (or downloaded from a Colab run - see docs/TRAINING.md).
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

# Where training/download_data.py puts the dataset.
DATA_DIR = PROJECT_ROOT / "data"

# Confidence threshold below which detections are discarded before display
# and storage. Chosen during evaluation (see docs/TRAINING.md): lower values
# flood the table with false positives, higher values miss weak faults.
DEFAULT_CONF_THRESHOLD = 0.40

# IoU threshold for Non-Maximum Suppression at inference time. Ultralytics
# default is 0.7 for NMS; 0.45 keeps overlapping duplicate boxes apart on
# panels where several faults sit close together.
DEFAULT_IOU_THRESHOLD = 0.45

# The 8 fault classes, exactly as in the dataset's data.yaml. The index in
# this list is the class id the model was trained on.
CLASS_NAMES: list[str] = [
    "MultiByPassed",
    "MultiDiode",
    "MultiHotSpot",
    "SingleByPassed",
    "SingleDiode",
    "SingleHotSpot",
    "StringOpenCircuit",
    "StringReversedPolarity",
]

# Image extensions accepted by the upload screen; everything else that the
# uploader allows is treated as video for the tracking pipeline.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
