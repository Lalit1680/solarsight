"""Video tracking pipeline (stretch goal).

Uses ByteTrack via `model.track(..., persist=True)` so a fault seen across
many frames keeps one stable track_id, then `dedupe_by_track` collapses the
per-frame boxes into one record per faulty panel.
"""
from __future__ import annotations

import cv2
import numpy as np

from src import config
from src.pipelines.detection import Detection, dedupe_by_track, filter_detections, results_to_detections


def iter_video_detections(
    video_path: str,
    model: object,
    conf: float = config.DEFAULT_CONF_THRESHOLD,
    iou: float = config.DEFAULT_IOU_THRESHOLD,
):
    """Yield (frame_index, annotated_frame_bgr, detections) for each frame.

    A generator so the Streamlit UI can show progress frame by frame instead
    of freezing on a long video. `persist=True` is essential: without it
    ByteTrack would treat every frame as a new video and every panel would
    get a new track id (breaking the count-once dedup).
    """
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break  # normal end of stream
            result = model.track(
                source=frame, conf=conf, iou=iou, persist=True, verbose=False
            )[0]
            detections = filter_detections(results_to_detections(result), min_conf=conf)
            yield frame_index, frame, detections
            frame_index += 1
    finally:
        capture.release()


def summarize_tracks(detections_by_frame: list[list[Detection]]) -> dict[int, Detection]:
    """Collapse per-frame detections into one best record per track id."""
    flat = [det for frame_dets in detections_by_frame for det in frame_dets]
    return dedupe_by_track(flat)


def read_first_frame(video_path: str) -> np.ndarray | None:
    """First frame (BGR) for a thumbnail, or None if the video is unreadable."""
    capture = cv2.VideoCapture(video_path)
    try:
        ok, frame = capture.read()
        return frame if ok else None
    finally:
        capture.release()
