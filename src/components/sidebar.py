"""Sidebar: site name, threshold sliders, model status, navigation."""
from __future__ import annotations

import streamlit as st

from src import config
from src.pipelines.detection import model_exists


def render_sidebar() -> tuple[str, float, float, str]:
    """Render sidebar controls and return (site_name, conf, iou, page).

    Model-status message lives here so it shows on every page: if best.pt is
    missing the user learns it immediately, not after clicking Upload.
    """
    with st.sidebar:
        st.title("SolarSight")
        st.caption("Thermal fault inspection for solar panels")

        page = st.radio(
            "Screen",
            ["Upload", "History"],
            help="Upload new media or browse past inspections.",
        )

        site_name = st.text_input(
            "Site name",
            value="Demo Site",
            help="Stored with the inspection so History can filter by site.",
        )

        conf = st.slider(
            "Confidence threshold",
            min_value=0.05,
            max_value=0.95,
            value=config.DEFAULT_CONF_THRESHOLD,
            step=0.05,
            help="Detections below this confidence are discarded.",
        )
        iou = st.slider(
            "IoU threshold (NMS)",
            min_value=0.20,
            max_value=0.80,
            value=config.DEFAULT_IOU_THRESHOLD,
            step=0.05,
            help="Higher = fewer duplicate boxes merged away.",
        )

        st.divider()
        if model_exists():
            st.success("Model loaded: models/best.pt")
        else:
            st.warning(
                "Model file `models/best.pt` not found.\n\n"
                "Train it (`python training/train_yolo.py`) or download it "
                "from the Colab run - see docs/TRAINING.md."
            )
    return site_name, conf, iou, page
