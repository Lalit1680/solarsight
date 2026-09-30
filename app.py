"""SolarSight - thermal solar-panel fault inspection.

Single-page Streamlit app. The sidebar picks the screen; everything heavy
(model loading) is cached so it happens once per process.
"""
import streamlit as st

from src.components.sidebar import render_sidebar

st.set_page_config(
    page_title="SolarSight",
    page_icon="☀️",
    layout="wide",
)

site_name, conf, iou, page = render_sidebar()

# Screens are imported lazily so the History screen never pays for loading the
# Upload screen's heavier imports. (The YOLO model itself is cached with
# st.cache_resource inside src/screens/upload.py.)
if page == "Upload":
    from src.screens.upload import render as render_upload

    render_upload(site_name, conf, iou)
else:
    from src.screens.history import render as render_history

    render_history()
