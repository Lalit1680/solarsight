"""Unit tests for the detection pipeline (all with fake model results)."""
from __future__ import annotations

import numpy as np

from src.pipelines.detection import (
    Detection,
    dedupe_by_track,
    filter_detections,
    model_exists,
    results_to_detections,
    run_detection,
)

from .conftest import FakeBoxes, FakeResult, NAMES, make_detection


# ---------------------------------------------------------------- conversion
def test_results_to_detections_converts_boxes(fake_result_single):
    dets = results_to_detections(fake_result_single)
    assert len(dets) == 1
    det = dets[0]
    assert det.class_index == 5
    assert det.class_name == "SingleHotSpot"
    assert det.confidence == 0.92
    # coordinates arrive as plain floats (JSON-safe), not np.float32
    assert det.x1 == 10.0 and det.y2 == 120.0
    assert isinstance(det.x1, float)
    assert det.track_id is None


def test_results_to_detections_handles_no_boxes(fake_result_empty):
    # ultralytics gives boxes=None when nothing found - must not crash
    assert results_to_detections(fake_result_empty) == []


def test_results_to_detections_keeps_track_ids_when_present():
    result = FakeResult(
        FakeBoxes([(0, 0, 10, 10, 0.8, 2)], ids=[7])
    )
    dets = results_to_detections(result)
    assert dets[0].track_id == 7


def test_plain_python_lists_also_work():
    """Robustness: numpy-free inputs (lists) are converted too."""
    result = FakeResult(
        FakeBoxes([(0, 0, 10, 10, 0.8, 2)])
    )
    assert results_to_detections(result)[0].class_name == "MultiHotSpot"


# ------------------------------------------------------------------ filtering
def test_filter_detections_keeps_only_above_threshold(fake_result_mixed):
    dets = results_to_detections(fake_result_mixed)
    kept = filter_detections(dets, min_conf=0.40)
    assert [d.confidence for d in kept] == [0.92, 0.75]
    assert all(d.confidence >= 0.40 for d in kept)


def test_filter_at_exactly_threshold_is_inclusive():
    det = make_detection(conf=0.50)
    assert filter_detections([det], min_conf=0.50) == [det]


def test_filter_empty_list_returns_empty():
    assert filter_detections([], min_conf=0.40) == []


# --------------------------------------------------------------------- dedupe
def test_dedupe_by_track_keeps_best_confidence_per_track():
    low = make_detection(conf=0.55, track_id=1)
    high = make_detection(conf=0.88, track_id=1)
    other = make_detection(conf=0.61, track_id=2)
    best = dedupe_by_track([low, high, other])
    assert set(best) == {1, 2}
    assert best[1].confidence == 0.88
    assert best[2].confidence == 0.61


def test_dedupe_ignores_frame_only_detections():
    no_track = make_detection(conf=0.99, track_id=None)
    assert dedupe_by_track([no_track]) == {}


# ------------------------------------------------------------------- run path
def test_run_detection_end_to_end_with_fake_model(fake_result_mixed, sample_image):
    class FakeModel:
        def predict(self, source, conf, iou, verbose):
            self.last_conf = conf
            return [fake_result_mixed]

    model = FakeModel()
    dets = run_detection(sample_image, model, conf=0.40, iou=0.45)
    assert model.last_conf == 0.40
    assert len(dets) == 2  # 0.35-confidence box filtered out


def test_model_exists_false_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.MODEL_PATH", tmp_path / "nope.pt")
    assert model_exists() is False


def test_detection_is_frozen_dataclass():
    det = make_detection()
    try:
        det.confidence = 0.1  # type: ignore[misc]
        raised = False
    except Exception:  # noqa: BLE001
        raised = True
    assert raised, "Detection should be immutable so results can't be mutated by mistake"
