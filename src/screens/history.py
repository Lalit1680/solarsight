"""History screen: past inspections, filters, charts, CSV export."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.components.export import inspections_to_dataframe, to_csv_bytes
from src.database.db import InspectionsDB


def _filter_controls(db: InspectionsDB) -> tuple[str | None, str | None]:
    """Sidebar-style filter row. Returns (site_name_or_None, media_type_or_None)."""
    col1, col2 = st.columns(2)
    try:
        sites = db.distinct_sites()
    except Exception:  # noqa: BLE001 - offline DB should show, not crash
        sites = []

    site = col1.selectbox(
        "Site", options=["All sites"] + sites, index=0,
        disabled=not sites,
        help="Populated from inspections already stored in Supabase.",
    )
    media = col2.selectbox("Media type", ["All", "image", "video"], index=0)
    return (
        None if site == "All sites" else site,
        None if media == "All" else media,
    )


def _inspections_dataframe(rows: list[dict]) -> pd.DataFrame:
    """DB rows -> display DataFrame (delegates to the pure export module)."""
    return inspections_to_dataframe(rows)


def render() -> None:
    st.header("History")
    st.caption("Past inspections stored in Supabase, with fault counts and CSV export.")

    from src.database.db import get_db, is_configured

    if not is_configured():
        st.warning(
            "History needs Supabase. Copy `.streamlit/secrets.toml.example` to "
            "`.streamlit/secrets.toml`, paste your project keys, run the SQL in "
            "`supabase/schema.sql`, then reload this page."
        )
        return

    try:
        db = get_db()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not connect to Supabase: {exc}")
        return

    site_filter, media_filter = _filter_controls(db)

    try:
        rows = db.list_inspections(site_name=site_filter, media_type=media_filter)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Query failed: {exc}")
        return

    st.markdown(f"### Inspections ({len(rows)})")
    st.dataframe(
        _inspections_dataframe(rows), use_container_width=True, hide_index=True
    )
    if not rows:
        st.info("No inspections yet - upload one on the Upload screen.")
        return

    # ---- Charts ----------------------------------------------------------
    counts = db.fault_counts(inspection_ids=[row["id"] for row in rows])
    chart_frame = pd.DataFrame(
        [
            {"class": class_name, "faults": counts.get(class_name, 0)}
            for class_name in config.CLASS_NAMES
        ]
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Fault counts by class (filtered)")
        st.bar_chart(chart_frame.set_index("class"))
    with right:
        st.subheader("Faults per inspection")
        per_inspection = pd.DataFrame(
            [{"when": row["created_at"][:16].replace("T", " "),
              "faults": row["total_faults"]} for row in rows]
        ).set_index("when")
        st.bar_chart(per_inspection)

    # ---- CSV export ------------------------------------------------------
    st.divider()
    st.subheader("Export")
    export_frame = _inspections_dataframe(rows)
    st.download_button(
        "Download CSV report",
        data=to_csv_bytes(export_frame),
        file_name="solarsight_report.csv",
        mime="text/csv",
        help="The inspection list exactly as filtered above.",
    )

    # ---- Detail drill-down ----------------------------------------------
    st.divider()
    st.subheader("Inspection detail")
    selected_id = st.selectbox(
        "Choose an inspection", options=[row["id"] for row in rows],
        format_func=lambda iid: next(
            (f"{r['site_name']} - {r['source_filename']}" for r in rows if r["id"] == iid),
            iid,
        ),
    )
    try:
        dets = db.get_detections(selected_id)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load detections: {exc}")
        return
    if dets:
        st.dataframe(pd.DataFrame(dets).drop(columns=["id", "inspection_id"]),
                     use_container_width=True, hide_index=True)
    else:
        st.info("This inspection has no stored detections (zero faults found).")
