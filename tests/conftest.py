"""Shared test fixtures: fake YOLO results and a fake Supabase client.

No torch, no network - that is the point. The fakes mimic exactly the small
surface of each library that our code touches, so tests fail for OUR bugs,
not for library version drift.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.pipelines.detection import Detection

NAMES = {
    0: "MultiByPassed",
    1: "MultiDiode",
    2: "MultiHotSpot",
    3: "SingleByPassed",
    4: "SingleDiode",
    5: "SingleHotSpot",
    6: "StringOpenCircuit",
    7: "StringReversedPolarity",
}


# --------------------------------------------------------------------------
# Fake ultralytics result objects (duck-typed: only .boxes/.names are used)
# --------------------------------------------------------------------------
class FakeBoxes:
    """Mimics ultralytics Boxes: torch-like tensors with .cpu().numpy()."""

    class _Tensor:
        def __init__(self, values):
            self._values = np.asarray(values)

        def cpu(self):
            return self

        def numpy(self):
            return self._values

    def __init__(self, rows, ids=None):
        # rows: list of (x1, y1, x2, y2, conf, cls)
        self.xyxy = self._Tensor([row[:4] for row in rows])
        self.conf = self._Tensor([row[4] for row in rows])
        self.cls = self._Tensor([row[5] for row in rows])
        self.id = self._Tensor(ids) if ids is not None else None


class FakeResult:
    def __init__(self, boxes, names=NAMES):
        self.boxes = boxes
        self.names = names


@pytest.fixture
def fake_result_single():
    """One hotspot at 0.92 confidence."""
    return FakeResult(
        FakeBoxes([(10.0, 20.0, 110.0, 120.0, 0.92, 5)])
    )


@pytest.fixture
def fake_result_mixed():
    """Three boxes with different confidences for filter tests."""
    return FakeResult(
        FakeBoxes(
            [
                (10.0, 20.0, 110.0, 120.0, 0.92, 5),   # SingleHotSpot - kept
                (200.0, 200.0, 260.0, 260.0, 0.35, 1), # MultiDiode    - dropped at 0.40
                (300.0, 300.0, 500.0, 500.0, 0.75, 3), # SingleByPassed - kept
            ]
        )
    )


@pytest.fixture
def fake_result_empty():
    """boxes=None: what ultralytics returns when nothing is detected."""
    return FakeResult(boxes=None)


def make_detection(conf=0.9, class_name="SingleHotSpot", track_id=None):
    return Detection(
        class_index=5,
        class_name=class_name,
        confidence=conf,
        x1=10.0,
        y1=10.0,
        x2=50.0,
        y2=50.0,
        track_id=track_id,
    )


# --------------------------------------------------------------------------
# Fake Supabase client: implements the exact fluent chain our db.py uses
# --------------------------------------------------------------------------
class FakeQuery:
    """Captures chained calls and returns canned data."""

    def __init__(self, store):
        self._store = store
        self._table = None
        self._op = None
        self._payload = None
        self._filters = []

    def table(self, name):
        self._table = name
        return self

    def insert(self, payload):
        self._op = "insert"
        self._payload = payload
        return self

    def delete(self):
        self._op = "delete"
        return self

    def select(self, _columns="*"):
        self._op = "select"
        return self

    def update(self, payload):
        self._op = "update"
        self._payload = payload
        return self

    def eq(self, column, value):
        self._filters.append(("eq", column, value))
        return self

    def in_(self, column, values):
        self._filters.append(("in", column, list(values)))
        return self

    def order(self, _column, desc=False):
        return self

    def limit(self, _n):
        return self

    def execute(self):
        if self._op == "insert":
            rows = self._payload if isinstance(self._payload, list) else [self._payload]
            inserted = []
            for row in rows:
                stored = dict(row)
                stored.setdefault("id", f"generated-{len(self._store[self._table]) + 1}")
                stored.setdefault("created_at", "2026-09-29T12:00:00.000Z")
                self._store[self._table].append(stored)
                inserted.append(stored)
            return type("R", (), {"data": inserted})()

        if self._op == "delete":
            before = len(self._store[self._table])
            self._store[self._table] = [
                row for row in self._store[self._table]
                if not all(
                    row.get(col) == val
                    for _, col, val in self._filters
                )
            ]
            deleted = before - len(self._store[self._table])
            return type("R", (), {"data": [{"deleted": deleted}]})()

        # select
        rows = list(self._store.get(self._table, []))
        for op, col, val in self._filters:
            if op == "in":
                rows = [row for row in rows if row.get(col) in val]
            else:  # eq
                rows = [row for row in rows if row.get(col) == val]
        return type("R", (), {"data": rows})()


class FakeSupabaseClient:
    """Backed by a plain dict of lists - inspectable in assertions."""

    def __init__(self):
        self.store = {"inspections": [], "detections": []}
        self.queries: list[FakeQuery] = []

    def table(self, name):
        query = FakeQuery(self.store)
        self.queries.append(query)
        return query.table(name)


@pytest.fixture
def fake_db_client():
    return FakeSupabaseClient()


@pytest.fixture
def db(fake_db_client):
    from src.database.db import InspectionsDB

    return InspectionsDB(fake_db_client)


@pytest.fixture
def sample_image() -> np.ndarray:
    """128x128 gray image with a bright square: enough for draw tests."""
    image = np.full((128, 128, 3), 60, dtype=np.uint8)
    image[40:80, 40:80] = 255
    return image
