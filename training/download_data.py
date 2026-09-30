"""Download the SolarSight dataset and verify it, end to end.

Dataset: "Thermal Solar PV Anomaly Detection Dataset" (ThermoSolar-PV)
  Kaggle:   https://www.kaggle.com/datasets/pkdarabi/solarpanel
  Licence:  CC BY-NC 4.0 (attribution, non-commercial) - fine for this
            portfolio project; do NOT sell anything built on it.
  Content:  7,339 thermal (infrared) images of solar panels, 640x640,
            YOLO format labels, 8 fault classes, pre-split
            train 6,924 / valid 400 / test 15.

Usage:
  python training/download_data.py            # download + verify, print stats

No Kaggle account or API key is needed: kagglehub downloads public datasets
anonymously (verified 2026-09-29). The download lands in a local kagglehub
cache and is then copied into ./data/ImageSet so training paths stay stable.

The verification step exists so nobody can accidentally train on a broken or
partial download: we re-count images/labels from the actual files and fail
loudly on any mismatch.
"""
from __future__ import annotations

import shutil
import sys
from collections import Counter
from pathlib import Path

# Allow `python training/download_data.py` from any cwd: add the project root
# (parent of training/) to the import path before touching src.*.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402  (path setup must happen first)

KAGGLE_HANDLE = "pkdarabi/solarpanel"

EXPECTED_SPLITS = {"train": 6924, "valid": 400, "test": 15}

# Class order matters: it must exactly match data.yaml, because the model's
# output indices are positions in this list.
EXPECTED_CLASSES = [
    "MultiByPassed",
    "MultiDiode",
    "MultiHotSpot",
    "SingleByPassed",
    "SingleDiode",
    "SingleHotSpot",
    "StringOpenCircuit",
    "StringReversedPolarity",
]


def download() -> Path:
    """Fetch via kagglehub and copy into ./data/ImageSet. Returns the target."""
    import kagglehub  # lazy: only needed for download, not for training itself

    cache_path = Path(kagglehub.dataset_download(KAGGLE_HANDLE))
    target = config.DATA_DIR / "ImageSet"
    if target.exists() and (target / "data.yaml").is_file():
        print(f"Already present: {target} (skipping copy)")
        return target

    print(f"Downloaded to cache: {cache_path}")
    # The kagglehub cache is <version>/ImageSet/... ; the archive's ImageSet
    # folder IS the dataset root (it holds data.yaml), so copy its CONTENTS
    # into data/ImageSet rather than the whole version folder - otherwise we
    # end up with data/ImageSet/ImageSet/ (double nesting).
    source = cache_path / "ImageSet"
    if not source.is_dir():
        source = cache_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True)
    print(f"Copied to: {target}")
    return target


def verify(root: Path) -> None:
    """Cross-check the downloaded files against the documented facts."""
    yaml_text = (root / "data.yaml").read_text()
    for class_name in EXPECTED_CLASSES:
        if class_name not in yaml_text:
            raise SystemExit(f"data.yaml is missing class {class_name!r} - dataset changed?")

    for split, expected_images in EXPECTED_SPLITS.items():
        images = sorted((root / split / "images").glob("*.jpg"))
        labels = sorted((root / split / "labels").glob("*.txt"))
        if len(images) != expected_images:
            raise SystemExit(
                f"{split}: expected {expected_images} images, found {len(images)}"
            )
        missing = {img.stem for img in images} - {lbl.stem for lbl in labels}
        if missing:
            raise SystemExit(f"{split}: {len(missing)} images have no label file")

    print("Verification passed: splits, label files and class list all match.")


def per_class_counts(root: Path) -> dict[str, Counter]:
    """Instance counts per class per split, read from the label files."""
    counts: dict[str, Counter] = {}
    for split in EXPECTED_SPLITS:
        counter: Counter = Counter()
        for label_file in (root / split / "labels").glob("*.txt"):
            for line in label_file.read_text().splitlines():
                parts = line.split()
                if parts:
                    counter[int(parts[0])] += 1
        counts[split] = counter
    return counts


def main() -> None:
    root = download()
    verify(root)

    print("\nPer-class bounding-box instances (from label files):")
    header = f"{'class':<24}" + "".join(f"{s:>8}" for s in EXPECTED_SPLITS)
    print(header)
    for index, class_name in enumerate(EXPECTED_CLASSES):
        row = f"{class_name:<24}"
        row += "".join(f"{per_class_counts(root)[split][index]:>8}" for split in EXPECTED_SPLITS)
        print(row)
    print("\nDone. Next: python training/train_yolo.py")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - report any failure clearly
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
