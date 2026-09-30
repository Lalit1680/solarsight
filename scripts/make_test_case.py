"""Build a visible test case: ground-truth answer key vs live model prediction.

Given a test-split image, produces two annotated PNGs in docs/assets/:

  * <stem>_truth.png - the image with GROUND-TRUTH boxes (green) and class
    names, i.e. the answer key drawn from the dataset labels.
  * <stem>_pred.png  - the same image with the trained model's PREDICTIONS
    (red) and confidences, using the app's default confidence threshold.

and prints an honest per-box comparison (IoU matching at 0.50, same class).

Usage:
    python scripts/make_test_case.py --name 124_jpg.rf.a50e9cd85618126df170f30070e282b7 --conf 0.40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_NAMES, DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD  # noqa: E402
from src.pipelines.detection import load_model, results_to_detections  # noqa: E402

GREEN = (43, 168, 74)   # BGR for ground truth
RED = (44, 54, 222)     # BGR for predictions
FONT = cv2.FONT_HERSHEY_SIMPLEX
IMG_DIR = PROJECT_ROOT / "data" / "ImageSet" / "test" / "images"
LBL_DIR = PROJECT_ROOT / "data" / "ImageSet" / "test" / "labels"
OUT_DIR = PROJECT_ROOT / "docs" / "assets"


def load_ground_truth(stem: str) -> list[dict]:
    """Parse the YOLO label file into pixel-space boxes."""
    rows = []
    label_file = LBL_DIR / f"{stem}.txt"
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, w, h = int(parts[0]), *(float(p) for p in parts[1:])
        x1, y1 = (cx - w / 2) * 640, (cy - h / 2) * 640
        x2, y2 = (cx + w / 2) * 640, (cy + h / 2) * 640
        rows.append({"cls": cls, "box": (x1, y1, x2, y2)})
    return rows


def draw(image, boxes: list[dict], color, prefix: str, with_conf: bool = False):
    for i, item in enumerate(boxes, 1):
        x1, y1, x2, y2 = (int(round(v)) for v in item["box"])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{prefix} {i}: {CLASS_NAMES[item['cls']]}"
        if with_conf:
            label += f" {item.get('conf', 0):.2f}"
        ty = y1 - 6 if y1 - 6 > 12 else y2 + 18
        cv2.putText(image, label, (x1, ty), FONT, 0.45, color, 1, cv2.LINE_AA)
    return image


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="test image stem (no extension)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD)
    args = parser.parse_args()

    img_path = IMG_DIR / f"{args.name}.jpg"
    if not img_path.exists():
        sys.exit(f"image not found: {img_path}")
    image = cv2.imread(str(img_path))
    if image is None:
        sys.exit(f"could not read: {img_path}")

    truth = load_ground_truth(args.name)

    model = load_model()
    raw = model.predict(image, conf=args.conf, iou=DEFAULT_IOU_THRESHOLD, verbose=False)[0]
    preds = [
        {"cls": CLASS_NAMES.index(d.class_name), "box": (d.x1, d.y1, d.x2, d.y2), "conf": d.confidence}
        for d in results_to_detections(raw)
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    truth_png = OUT_DIR / f"test_case_{args.name.split('_jpg')[0]}_truth.png"
    pred_png = OUT_DIR / f"test_case_{args.name.split('_jpg')[0]}_pred.png"
    cv2.imwrite(str(truth_png), draw(image.copy(), truth, GREEN, "GT"))
    cv2.imwrite(str(pred_png), draw(image.copy(), preds, RED, "Pred", with_conf=True))

    # honest comparison: match each GT box to the best same-class prediction
    print(f"\nImage: {img_path.name}  (conf>={args.conf}, IoU match>=0.50)")
    print(f"Ground-truth faults: {len(truth)} | Model predictions: {len(preds)}\n")
    tp = 0
    for gi, gt in enumerate(truth, 1):
        best, best_iou = None, 0.0
        for pi, pr in enumerate(preds):
            if pr["cls"] != gt["cls"]:
                continue
            overlap = iou(gt["box"], pr["box"])
            if overlap > best_iou:
                best, best_iou = pi, overlap
        name = CLASS_NAMES[gt["cls"]]
        if best is not None and best_iou >= 0.50:
            tp += 1
            print(f"  GT {gi} {name:16s} -> FOUND (Pred {best+1}, IoU {best_iou:.2f}, conf {preds[best]['conf']:.2f})")
        elif best is not None:
            print(f"  GT {gi} {name:16s} -> WRONG BOX (best same-class IoU {best_iou:.2f} < 0.50)")
        else:
            print(f"  GT {gi} {name:16s} -> MISSED")
    fp = len(preds) - tp
    print(f"\nResult: {tp}/{len(truth)} found | {fp} false alarm(s) | truth png: {truth_png.name} | pred png: {pred_png.name}")


if __name__ == "__main__":
    main()
