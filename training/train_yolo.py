"""Train the YOLO detector on the thermal solar dataset.

Usage:
  python training/train_yolo.py                 # sensible defaults
  python training/train_yolo.py --epochs 60     # longer run on GPU
  python training/train_yolo.py --device mps    # Apple silicon GPU

What it does:
  * Prints which device it will use (GPU/CPU) - never silently train on CPU
    when you expected a GPU.
  * Trains YOLO11-nano, the smallest YOLO variant: fastest to fine-tune and
    small enough (weights < 6 MB) to keep the app responsive on CPU.
  * Writes runs/solar/train/weights/best.pt; copy it to models/best.pt when
    you are happy with it (or let --export-model do it automatically).

Hyperparameters are documented in docs/TRAINING.md together with the real
measured results of the run that produced the shipped model.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402  (path setup must happen first)

# Image size 640: the dataset ships pre-resized to 640x640, so no further
# resizing information is thrown away at train time.
IMG_SIZE = 640


def describe_device(requested: str | None) -> str:
    """Pick the device the same way ultralytics will, and SAY it out loud."""
    import torch  # lazy: only needed when actually training

    if requested:
        return requested
    if torch.cuda.is_available():
        return "0"  # first CUDA GPU
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs (30 is enough for a fine-tune; 60+ on GPU).")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size; lower it if you hit out-of-memory.")
    parser.add_argument("--device", type=str, default=None,
                        help="Force a device: 'cpu', '0' (first GPU) or 'mps'.")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="Starting checkpoint (COCO-pretrained nano by default).")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Training image size. The dataset is 640x640; on a CPU "
                             "a reduced size (e.g. 320) keeps epochs manageable at "
                             "some cost in small-defect recall.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Dataloader worker processes.")
    parser.add_argument("--export-model", action="store_true",
                        help="Copy the best weights to models/best.pt at the end.")
    args = parser.parse_args()

    data_root = config.DATA_DIR / "ImageSet"
    data_yaml = data_root / "data.yaml"
    if not data_yaml.is_file():
        raise SystemExit(
            f"No dataset at {data_root}. Run: python training/download_data.py"
        )

    device = describe_device(args.device)
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: training on CPU. Fine-tuning yolo11n for 30 epochs takes"
              " roughly 30-60 min on 8 cores. For faster runs use Google Colab"
              " (free T4 GPU) - see docs/TRAINING.md.")

    from ultralytics import YOLO  # lazy import so --help stays fast

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project="runs/solar",
        name="train",
        # Reproducibility: fixed seed makes the split shuffling deterministic.
        seed=42,
        # Small dataset (6.9k images): moderate augmentation. Mosaic is on by
        # default and helps small hotspots; we disable flipping vertically
        # because panels in drone footage have a natural 'up'.
        fliplr=0.5,
        flipud=0.0,
        # Keep the last and the best checkpoint.
        save_period=0,
        patience=15,  # early-stop if validation does not improve for 15 epochs
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nBest weights: {best}")
    if args.export_model:
        config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, config.MODEL_PATH)
        print(f"Copied to {config.MODEL_PATH} - the app will use these weights.")
    else:
        print("Copy it to models/best.pt (or re-run with --export-model).")


if __name__ == "__main__":
    main()
