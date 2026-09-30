"""Draw detection boxes on an image.

Kept as pure functions (numpy in, numpy out) so they can be unit-tested
without running Streamlit or loading a model.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from src.pipelines.detection import Detection

# Fixed palette indexed by class id: stable colors across runs make the demo
# easier to explain ("hotspot is always orange"). 8 entries cover the 8 classes.
PALETTE: list[tuple[int, int, int]] = [
    (255, 99, 71),    # MultiByPassed        - tomato
    (255, 215, 0),    # MultiDiode           - gold
    (255, 140, 0),    # MultiHotSpot         - dark orange
    (30, 144, 255),   # SingleByPassed       - dodger blue
    (0, 200, 83),     # SingleDiode          - green
    (255, 49, 49),    # SingleHotSpot        - red
    (170, 0, 255),    # StringOpenCircuit    - purple
    (0, 206, 209),    # StringReversedPolarity - turquoise
]


def color_for_class_index(class_index: int) -> tuple[int, int, int]:
    """Deterministic color for a class id, wrapping around the palette."""
    return PALETTE[class_index % len(PALETTE)]


def draw_detections(
    image_rgb: np.ndarray,
    detections: list[Detection],
    class_names: list[str] | None = None,
) -> Image.Image:
    """Return a copy of `image_rgb` (numpy HxWx3, RGB) with boxes + labels drawn.

    We draw with PIL because its `textbbox` makes readable labels trivial.
    A copy is drawn on so the caller's original frame stays clean.
    """
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)

    for det in detections:
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        class_idx = det.class_index
        color = color_for_class_index(class_idx)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = det.class_name
        if class_names is not None and class_idx < len(class_names):
            label = f"{class_names[class_idx]} {det.confidence:.2f}"
        else:
            label = f"{label} {det.confidence:.2f}"
        if det.track_id is not None:
            label = f"#{det.track_id} {label}"

        # Label plate above the box; clamped inside the image so it never
        # disappears when a box touches the top edge.
        tb = draw.textbbox((0, 0), label)
        text_w, text_h = tb[2] - tb[0], tb[3] - tb[1]
        plate_y = max(0, y1 - text_h - 6)
        draw.rectangle([x1, plate_y, x1 + text_w + 6, plate_y + text_h + 4], fill=color)
        draw.text((x1 + 3, plate_y + 2), label, fill=(0, 0, 0))

    return pil_image


def count_by_class(detections: list[Detection]) -> dict[str, int]:
    """Fault counts keyed by class name - feeds the metrics row and charts."""
    counts: dict[str, int] = {}
    for det in detections:
        counts[det.class_name] = counts.get(det.class_name, 0) + 1
    return counts
