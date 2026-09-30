"""Results table + metrics components for the Upload screen."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.pipelines.detection import Detection


def detections_to_dataframe(detections: list[Detection]) -> pd.DataFrame:
    """Detections -> display DataFrame. Pure function so it is testable."""
    if not detections:
        return pd.DataFrame(
            columns=["class", "confidence", "x1", "y1", "x2", "y2", "track_id"]
        )
    return pd.DataFrame(
        [
            {
                "class": d.class_name,
                "confidence": round(d.confidence, 3),
                "x1": round(d.x1, 1),
                "y1": round(d.y1, 1),
                "x2": round(d.x2, 1),
                "y2": round(d.y2, 1),
                "track_id": d.track_id,
            }
            for d in detections
        ]
    )


def render_results_table(detections: list[Detection]) -> None:
    """Metrics row + sortable results table, with an explicit empty state."""
    counts: dict[str, int] = {}
    for det in detections:
        counts[det.class_name] = counts.get(det.class_name, 0) + 1

    st.markdown(f"### Results - {len(detections)} fault(s) found")
    if not detections:
        st.info("No faults found above the confidence threshold. "
                "Try lowering the threshold in the sidebar.")
        return

    metric_cols = st.columns(min(len(counts), 4))
    for col, (class_name, count) in zip(metric_cols, counts.items()):
        col.metric(class_name, count)
    # If there are more than 4 distinct classes, show the rest in a second row.
    remaining = list(counts.items())[4:]
    if remaining:
        extra_cols = st.columns(min(len(remaining), 4))
        for col, (class_name, count) in zip(extra_cols, remaining):
            col.metric(class_name, count)

    st.dataframe(
        detections_to_dataframe(detections),
        use_container_width=True,
        hide_index=True,
    )
