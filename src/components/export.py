"""CSV export helpers for the History screen.

Extracted into a pure module so the export logic is unit-testable without
Streamlit and reusable for future report formats.
"""
from __future__ import annotations

import pandas as pd

COLUMNS = ["created_at", "site", "file", "media", "model", "faults", "id"]


def inspections_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    """DB rows -> display/exports DataFrame with inspector-friendly columns."""
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame(
        [
            {
                "created_at": row["created_at"][:19].replace("T", " "),
                "site": row["site_name"],
                "file": row["source_filename"],
                "media": row["media_type"],
                "model": row["model_version"],
                "faults": row["total_faults"],
                "id": row["id"],
            }
            for row in rows
        ]
    )


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """DataFrame -> CSV file bytes for st.download_button."""
    return frame.to_csv(index=False).encode("utf-8")
