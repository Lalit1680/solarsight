"""Detection pipeline.

All model-facing logic lives here. `ultralytics` is imported LAZILY inside
functions so that importing this module (e.g. from pytest or CI) never pulls
in torch. Tests use small fake result objects instead of real model output.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src import config


@dataclass(frozen=True)
class Detection:
    """One fault found in one image/video frame, ready for UI and database.

    Coordinates are absolute pixels (x1, y1 = top-left, x2, y2 = bottom-right),
    which is what both the drawing code and the database rows need. The YOLO
    output (normalized center format) is converted in `results_to_detections`.
    """

    class_index: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int | None = None  # set only in video mode (ByteTrack)


def filter_detections(detections: list[Detection], min_conf: float) -> list[Detection]:
    """Keep detections with confidence >= min_conf.

    >= (not >) so that a configured threshold of e.g. 0.5 keeps a detection
    that came back with exactly 0.5 - the intuitive reading of "minimum".
    """
    return [d for d in detections if d.confidence >= min_conf]


def results_to_detections(result: object) -> list[Detection]:
    """Convert one ultralytics `Results` object into Detection records.

    We duck-type the input (needs `.boxes` with `.xyxy/.conf/.cls` and a
    `.names` dict) instead of importing ultralytics here - that keeps this
    module importable without torch and makes the function trivially testable
    with lightweight fakes.

    Handles the empty case: ultralytics returns `boxes=None` when nothing was
    found, which would crash naive `len(boxes.xyxy)` code.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    names: dict[int, str] = result.names  # type: ignore[attr-defined]

    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
    classes = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)

    # track_ids is None for plain predict(); present after model.track().
    track_ids = getattr(boxes, "id", None)
    if track_ids is not None:
        track_ids = track_ids.cpu().numpy() if hasattr(track_ids, "cpu") else np.asarray(track_ids)

    detections: list[Detection] = []
    for i in range(len(xyxy)):
        class_index = int(classes[i])
        track_id = int(track_ids[i]) if track_ids is not None else None
        detections.append(
            Detection(
                class_index=class_index,
                class_name=names[class_index],
                confidence=float(confs[i]),
                # np.float32 is not JSON-serializable -> cast to float here so
                # the same objects can go straight into Supabase later.
                x1=float(xyxy[i][0]),
                y1=float(xyxy[i][1]),
                x2=float(xyxy[i][2]),
                y2=float(xyxy[i][3]),
                track_id=track_id,
            )
        )
    return detections


def run_detection(
    image_bgr: np.ndarray,
    model: object,
    conf: float = config.DEFAULT_CONF_THRESHOLD,
    iou: float = config.DEFAULT_IOU_THRESHOLD,
) -> list[Detection]:
    """Run the model on one image (numpy BGR, as OpenCV/ultralytics expect).

    Returns filtered detections. `conf`/`iou` are passed to the model so NMS
    happens once inside ultralytics; `filter_detections` afterwards is a belt-
    and-braces guard for callers that build detections another way.
    """
    result = model.predict(source=image_bgr, conf=conf, iou=iou, verbose=False)[0]
    return filter_detections(results_to_detections(result), min_conf=conf)


def dedupe_by_track(detections: list[Detection]) -> dict[int, Detection]:
    """Video mode: one record per track_id, keeping each fault's BEST
    (highest-confidence) appearance across all frames.

    This is what turns per-frame boxes into "each faulty panel counted once".
    Frames without track ids are ignored - they are handled per-frame instead.
    """
    best: dict[int, Detection] = {}
    for det in detections:
        if det.track_id is None:
            continue
        current = best.get(det.track_id)
        if current is None or det.confidence > current.confidence:
            best[det.track_id] = det
    return best


def model_exists() -> bool:
    """True when trained weights are on disk. The UI uses this to show a
    friendly setup message instead of crashing deep inside ultralytics."""
    return config.MODEL_PATH.is_file()


def load_model(weights_path: str | None = None) -> object:
    """Load YOLO weights. The ONLY place that imports ultralytics.

    Lazy import on purpose: importing ultralytics imports torch (hundreds of
    MB, slow). Tests and CI import this module but never call load_model.
    """
    from ultralytics import YOLO  # noqa: PLC0415 - deliberate lazy import

    path = str(weights_path) if weights_path else str(config.MODEL_PATH)
    return YOLO(path)
