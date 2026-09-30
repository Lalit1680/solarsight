"""Unit tests for the database layer with the in-memory fake client."""
from __future__ import annotations

import pytest

from src.database.db import (
    InspectionsDB,
    detection_row,
    save_inspection,
)

from .conftest import make_detection


# --------------------------------------------------------------- row mapping
def test_detection_row_maps_all_fields():
    det = make_detection(0.75, track_id=3)
    row = detection_row(det)
    assert row == {
        "class_name": "SingleHotSpot",
        "confidence": 0.75,
        "x1": 10.0,
        "y1": 10.0,
        "x2": 50.0,
        "y2": 50.0,
        "track_id": 3,
    }
    # everything must be JSON-serializable for PostgREST
    import json

    json.dumps(row)


def test_detection_row_null_track_for_images():
    assert detection_row(make_detection(track_id=None))["track_id"] is None


# ------------------------------------------------------------------- inserts
def test_insert_inspection_returns_generated_id(db):
    inspection_id = db.insert_inspection(
        site_name="Demo Site",
        source_filename="panel.jpg",
        media_type="image",
        model_version="yolo:best.pt",
        total_faults=2,
    )
    assert inspection_id.startswith("generated-")
    stored = db._client.store["inspections"][0]
    assert stored["site_name"] == "Demo Site"
    assert stored["total_faults"] == 2


def test_insert_detections_empty_is_noop(db):
    inspection_id = db.insert_inspection(
        site_name="s", source_filename="f.jpg", media_type="image",
        model_version="m", total_faults=0,
    )
    assert db.insert_detections(inspection_id, []) == 0
    assert db._client.store["detections"] == []


def test_insert_detections_attaches_inspection_id(db):
    inspection_id = db.insert_inspection(
        site_name="s", source_filename="f.jpg", media_type="image",
        model_version="m", total_faults=1,
    )
    count = db.insert_detections(inspection_id, [detection_row(make_detection())])
    assert count == 1
    assert db._client.store["detections"][0]["inspection_id"] == inspection_id


def test_save_inspection_writes_parent_and_children(db):
    inspection_id = save_inspection(
        db=db,
        site_name="Rooftop A",
        source_filename="drone_001.jpg",
        media_type="image",
        model_version="yolo:best.pt",
        detections=[make_detection(0.9), make_detection(0.7)],
    )
    assert len(db._client.store["inspections"]) == 1
    assert len(db._client.store["detections"]) == 2
    parent = db._client.store["inspections"][0]
    assert parent["total_faults"] == 2
    assert all(
        row["inspection_id"] == inspection_id
        for row in db._client.store["detections"]
    )


# --------------------------------------------------------------------- reads
def test_list_inspections_orders_and_filters(db):
    db.insert_inspection(site_name="A", source_filename="a.jpg",
                         media_type="image", model_version="m", total_faults=1)
    db.insert_inspection(site_name="B", source_filename="b.mp4",
                         media_type="video", model_version="m", total_faults=3)
    db.insert_inspection(site_name="A", source_filename="a2.jpg",
                         media_type="image", model_version="m", total_faults=0)

    assert len(db.list_inspections()) == 3
    only_a = db.list_inspections(site_name="A")
    assert len(only_a) == 2 and all(r["site_name"] == "A" for r in only_a)
    only_video = db.list_inspections(media_type="video")
    assert len(only_video) == 1 and only_video[0]["source_filename"] == "b.mp4"


def test_get_detections_scopes_by_inspection(db):
    id1 = db.insert_inspection(site_name="s", source_filename="1.jpg",
                               media_type="image", model_version="m", total_faults=1)
    id2 = db.insert_inspection(site_name="s", source_filename="2.jpg",
                               media_type="image", model_version="m", total_faults=1)
    db.insert_detections(id1, [detection_row(make_detection())])
    db.insert_detections(id2, [detection_row(make_detection())])

    assert len(db.get_detections(id1)) == 1
    assert db.get_detections(id1)[0]["inspection_id"] == id1


def test_fault_counts_aggregates_by_class(db):
    id1 = db.insert_inspection(site_name="s", source_filename="1.jpg",
                               media_type="image", model_version="m", total_faults=2)
    db.insert_detections(
        id1,
        [
            detection_row(make_detection(class_name="SingleHotSpot")),
            detection_row(make_detection(class_name="SingleHotSpot")),
            detection_row(make_detection(class_name="MultiDiode")),
        ],
    )
    counts = db.fault_counts(inspection_ids=[id1])
    assert counts == {"SingleHotSpot": 2, "MultiDiode": 1}


def test_fault_counts_empty_ids_returns_empty(db):
    assert db.fault_counts(inspection_ids=[]) == {}


def test_delete_inspection_removes_parent(db):
    inspection_id = db.insert_inspection(site_name="s", source_filename="x.jpg",
                                         media_type="image", model_version="m",
                                         total_faults=0)
    db.delete_inspection(inspection_id)
    assert db._client.store["inspections"] == []


def test_distinct_sites_sorted_unique(db):
    for site in ["B", "A", "A", "C"]:
        db.insert_inspection(site_name=site, source_filename="f.jpg",
                             media_type="image", model_version="m", total_faults=0)
    assert db.distinct_sites() == ["A", "B", "C"]
