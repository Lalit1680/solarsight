"""Upload screen: image or video in, detections out, optional Supabase save.

Runs the model only here (and in the video pipeline), so everything else in
the app imports without touching torch.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from src import config
from src.components.results_table import render_results_table
from src.pipelines import video as video_pipeline
from src.pipelines.detection import Detection, load_model, run_detection
from src.ui.draw import draw_detections


@st.cache_resource(show_spinner="Loading YOLO model...")
def _load_model_cached() -> object:
    """Load weights once per process. cache_resource shares the model across
    user sessions; it is read-only at inference time so sharing is safe."""
    return load_model()


def _read_uploaded_image(file_bytes: bytes) -> np.ndarray:
    """Bytes -> numpy RGB. PIL is used so EXIF-rotated phone photos behave."""
    import io

    from PIL import Image

    pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.asarray(pil_image)


def _model_version(model: object) -> str:
    """Human-readable model version string stored with each inspection."""
    weights = getattr(model, "ckpt_path", None) or str(config.MODEL_PATH.name)
    return f"yolo:{Path(str(weights)).name}"


def render(site_name: str, conf: float, iou: float) -> None:
    st.header("Upload inspection")
    st.caption("Thermal images (jpg/png) or drone video (mp4/mov) of solar panels.")

    if not config.MODEL_PATH.is_file():
        st.error(
            "**Model not found** - `models/best.pt` is missing. "
            "Train it with `python training/train_yolo.py` (see docs/TRAINING.md), "
            "then restart the app. Nothing else in the app is affected."
        )
        return

    media_kind = st.radio("Media type", ["Image", "Video"], horizontal=True)
    extensions = (
        config.IMAGE_EXTENSIONS if media_kind == "Image" else config.VIDEO_EXTENSIONS
    )
    uploaded = st.file_uploader(
        f"Drop a {media_kind.lower()}",
        type=sorted({ext.lstrip(".") for ext in extensions}),
    )
    if uploaded is None:
        return

    model = _load_model_cached()
    file_bytes = uploaded.getvalue()

    if media_kind == "Image":
        _render_image_flow(model, uploaded.name, file_bytes, site_name, conf, iou)
    else:
        _render_video_flow(model, uploaded.name, file_bytes, site_name, conf, iou)


def _render_image_flow(
    model: object,
    filename: str,
    file_bytes: bytes,
    site_name: str,
    conf: float,
    iou: float,
) -> None:
    image_rgb = _read_uploaded_image(file_bytes)
    detections = run_detection(image_bgr=image_rgb[:, :, ::-1], model=model,
                               conf=conf, iou=iou)  # ultralytics expects BGR

    annotated = draw_detections(image_rgb, detections)
    left, right = st.columns(2)
    with left:
        st.image(image_rgb, caption="Uploaded", use_container_width=True)
    with right:
        st.image(annotated, caption="Detections", use_container_width=True)

    render_results_table(detections)

    _save_section(
        site_name=site_name,
        source_filename=filename,
        media_type="image",
        model_version=_model_version(model),
        detections=detections,
    )


def _render_video_flow(
    model: object,
    filename: str,
    file_bytes: bytes,
    site_name: str,
    conf: float,
    iou: float,
) -> None:
    """Write the upload to a temp file (OpenCV needs a path), track frames,
    show progress, then deduplicate by track_id so each faulty panel is
    counted once."""
    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        video_path = tmp.name

    progress = st.progress(0.0, text="Tracking frames...")
    per_frame: list[list[Detection]] = []
    preview = None
    total_frames: float | None = None

    try:
        import cv2

        capture = cv2.VideoCapture(video_path)
        total_frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()

        for frame_index, frame, detections in video_pipeline.iter_video_detections(
            video_path, model, conf=conf, iou=iou
        ):
            per_frame.append(detections)
            preview = (frame_index, frame, detections)
            if total_frames:
                progress.progress(
                    min(1.0, (frame_index + 1) / total_frames),
                    text=f"Tracking frames... {frame_index + 1}/{int(total_frames)}",
                )
    finally:
        Path(video_path).unlink(missing_ok=True)

    progress.progress(1.0, text="Tracking complete")

    if not per_frame:
        st.warning("No readable frames in that video.")
        return

    # One record per track: the best-confidence appearance of each panel.
    unique_faults = video_pipeline.summarize_tracks(per_frame)
    detections = list(unique_faults.values())

    st.markdown(f"### Video results - {len(detections)} unique faulty panel(s)")
    if preview is not None:
        frame_index, frame, frame_dets = preview
        annotated = draw_detections(frame[:, :, ::-1], frame_dets)
        st.image(annotated, caption=f"Last analysed frame #{frame_index}",
                 use_container_width=True)

    render_results_table(detections)
    _save_section(
        site_name=site_name,
        source_filename=filename,
        media_type="video",
        model_version=_model_version(model),
        detections=detections,
    )


def _save_section(
    site_name: str,
    source_filename: str,
    media_type: str,
    model_version: str,
    detections: list[Detection],
) -> None:
    """Save-to-Supabase block. Missing secrets disable saving with a hint,
    never a crash - the app is fully usable without a database."""
    from src.database.db import DBNotConfiguredError, get_db, is_configured, save_inspection

    st.divider()
    if not is_configured():
        st.info(
            "Saving is disabled: Supabase secrets are not configured. "
            "See `.streamlit/secrets.toml.example`."
        )
        return

    if st.button("Save inspection to Supabase", type="primary"):
        try:
            with st.spinner("Saving..."):
                db = get_db()
                inspection_id = save_inspection(
                    db=db,
                    site_name=site_name,
                    source_filename=source_filename,
                    media_type=media_type,
                    model_version=model_version,
                    detections=detections,
                )
            st.success(f"Saved inspection `{inspection_id}` "
                       f"({len(detections)} detections).")
        except DBNotConfiguredError as exc:
            st.error(f"Cannot save: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface Supabase errors to the inspector
            st.error(f"Save failed: {exc}")
