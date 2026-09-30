"""Tests for CSV export, drawing and the video summariser."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from src.components.export import inspections_to_dataframe, to_csv_bytes
from src.pipelines.detection import Detection
from src.pipelines.video import summarize_tracks
from src.ui.draw import color_for_class_index, count_by_class, draw_detections

from .conftest import make_detection


# ----------------------------------------------------------------- CSV export
def _row(**overrides):
    base = {
        "created_at": "2026-09-29T12:34:56.789Z",
        "site_name": "Rooftop A",
        "source_filename": "panel.jpg",
        "media_type": "image",
        "model_version": "yolo:best.pt",
        "total_faults": 3,
        "id": "abc-123",
    }
    base.update(overrides)
    return base


def test_inspections_to_dataframe_shapes_columns():
    frame = inspections_to_dataframe([_row()])
    assert list(frame.columns) == [
        "created_at", "site", "file", "media", "model", "faults", "id",
    ]
    assert frame.iloc[0]["site"] == "Rooftop A"
    assert frame.iloc[0]["created_at"] == "2026-09-29 12:34:56"
    assert frame.iloc[0]["faults"] == 3


def test_inspections_to_dataframe_empty():
    frame = inspections_to_dataframe([])
    assert frame.empty
    assert list(frame.columns) == [
        "created_at", "site", "file", "media", "model", "faults", "id",
    ]


def test_to_csv_bytes_round_trip():
    frame = inspections_to_dataframe([_row(), _row(site_name="Site B")])
    raw = to_csv_bytes(frame)
    assert isinstance(raw, bytes)

    back = pd.read_csv(io.BytesIO(raw))
    assert len(back) == 2
    assert back.iloc[1]["site"] == "Site B"
    assert list(back.columns)[0] == "created_at"


# --------------------------------------------------------------------- draw
def test_draw_detections_returns_image_and_leaves_original(sample_image):
    det = Detection(  # coordinates inside the 128x128 sample
        class_index=5, class_name="SingleHotSpot", confidence=0.9,
        x1=10, y1=10, x2=60, y2=60, track_id=None,
    )
    annotated = draw_detections(sample_image, [det])
    assert annotated.size == (128, 128)
    # original untouched (we draw on a copy)
    assert np.asarray(annotated)[45, 45].tolist() != [60, 60, 60]


def test_draw_with_track_label_and_empty_list(sample_image):
    assert draw_detections(sample_image, []).size == (128, 128)
    det = Detection(5, "SingleHotSpot", 0.8, 10, 10, 60, 60, track_id=4)
    assert draw_detections(sample_image, [det]).size == (128, 128)


def test_count_by_class_and_palette():
    dets = [
        make_detection(class_name="SingleHotSpot"),
        make_detection(class_name="SingleHotSpot"),
        make_detection(class_name="MultiDiode"),
    ]
    assert count_by_class(dets) == {"SingleHotSpot": 2, "MultiDiode": 1}
    # class index wraps deterministically instead of crashing
    assert color_for_class_index(0) == color_for_class_index(8)


# ------------------------------------------------------------------- tracking
def test_summarize_tracks_counts_each_panel_once():
    # Same panel (track 1) appears in 3 frames with rising confidence,
    # a second panel (track 2) in one frame.
    frames = [
        [make_detection(0.55, track_id=1)],
        [make_detection(0.75, track_id=1), make_detection(0.66, track_id=2)],
        [make_detection(0.91, track_id=1)],
    ]
    summary = summarize_tracks(frames)
    assert len(summary) == 2          # two unique faulty panels
    assert summary[1].confidence == 0.91  # best appearance kept
    assert summary[2].confidence == 0.66
