"""Run the trained model on held-out test images and save annotated outputs.

This is the "prove it works on real held-out data" step: it uses images the
model never trained on (the dataset's test split), draws the predictions,
and writes them to docs/assets/samples/ so the README can show real output.

Usage:
  python scripts/run_samples.py [--count 4] [--conf 0.40]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import config
from src.pipelines.detection import load_model, run_detection
from src.ui.draw import draw_detections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=4,
                        help="How many test images to run.")
    parser.add_argument("--conf", type=float, default=config.DEFAULT_CONF_THRESHOLD)
    args = parser.parse_args()

    if not config.MODEL_PATH.is_file():
        raise SystemExit(f"No model at {config.MODEL_PATH} - train first.")

    test_images = sorted((config.DATA_DIR / "ImageSet" / "test" / "images").glob("*.jpg"))
    if not test_images:
        raise SystemExit("No test images - run training/download_data.py first.")

    out_dir = config.PROJECT_ROOT / "docs" / "assets" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model()
    print(f"Model: {config.MODEL_PATH.name} | conf>={args.conf} | "
          f"{min(args.count, len(test_images))} test images\n")

    for image_path in test_images[: args.count]:
        from PIL import Image

        image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
        detections = run_detection(image_bgr=image_rgb[:, :, ::-1], model=model,
                                   conf=args.conf)
        annotated = draw_detections(image_rgb, detections)
        target = out_dir / f"{image_path.stem}_pred.jpg"
        annotated.save(target, quality=90)

        print(f"{image_path.name}: {len(detections)} fault(s)")
        for det in detections:
            print(f"   {det.class_name:<22} conf={det.confidence:.2f} "
                  f"box=({det.x1:.0f},{det.y1:.0f},{det.x2:.0f},{det.y2:.0f})")
        print(f"   -> {target.relative_to(config.PROJECT_ROOT)}")

    print("\nDone. These are real predictions on held-out images.")


if __name__ == "__main__":
    main()
