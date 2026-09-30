"""Database access. ALL SQL / Supabase calls live in this file.

Design: the class takes an injected `client` so tests can pass a fake that
mimics the supabase-py fluent API (`.table().insert().execute()` chains)
without any network. In production `get_db()` builds the real client from
Streamlit secrets.
"""
from __future__ import annotations

from typing import Any

from src import config


class DBNotConfiguredError(RuntimeError):
    """Raised when secrets.toml is missing or incomplete."""


def _client_from_secrets() -> Any:
    """Build a Supabase client from Streamlit secrets.

    Streamlit is imported INSIDE the function: db.py must stay importable in
    plain Python (tests, CI) where there is no Streamlit runtime and no
    secrets file.
    """
    import streamlit as st  # noqa: PLC0415 - lazy: not available under pytest
    from supabase import create_client  # noqa: PLC0415 - lazy: heavy import

    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    except (KeyError, FileNotFoundError) as exc:
        raise DBNotConfiguredError(
            "Supabase secrets are missing. Copy .streamlit/secrets.toml.example "
            "to .streamlit/secrets.toml and paste your project URL and anon key."
        ) from exc
    return create_client(url, key)


def get_db() -> "InspectionsDB":
    """Default entry point for the app: DB backed by real Supabase secrets."""
    return InspectionsDB(_client_from_secrets())


def is_configured() -> bool:
    """True when the secrets file has both required keys. Used by the UI to
    decide whether saving/history is possible - never raises."""
    try:
        _client_from_secrets()
        return True
    except Exception:  # noqa: BLE001 - any failure means "not configured"
        return False


class InspectionsDB:
    """Data access for the inspections + detections tables."""

    def __init__(self, client: Any) -> None:
        self._client = client

    # ------------------------------------------------------------- writes
    def insert_inspection(
        self,
        site_name: str,
        source_filename: str,
        media_type: str,
        model_version: str,
        total_faults: int,
    ) -> str:
        """Insert the parent row and return its generated UUID."""
        response = (
            self._client.table("inspections")
            .insert(
                {
                    "site_name": site_name,
                    "source_filename": source_filename,
                    "media_type": media_type,
                    "model_version": model_version,
                    "total_faults": total_faults,
                }
            )
            .execute()
        )
        return response.data[0]["id"]

    def insert_detections(self, inspection_id: str, rows: list[dict[str, Any]]) -> int:
        """Insert detection rows for an inspection. Returns the row count.

        Empty list is a valid no-op: an inspection can legitimately find zero
        faults, and PostgREST rejects empty bulk inserts.
        """
        if not rows:
            return 0
        payload = [{**row, "inspection_id": inspection_id} for row in rows]
        self._client.table("detections").insert(payload).execute()
        return len(payload)

    def delete_inspection(self, inspection_id: str) -> None:
        # detections rows disappear via ON DELETE CASCADE in schema.sql.
        self._client.table("inspections").delete().eq("id", inspection_id).execute()

    # -------------------------------------------------------------- reads
    def list_inspections(
        self,
        site_name: str | None = None,
        media_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Newest-first inspections, optionally filtered. Filters are applied
        in the DATABASE (PostgREST .eq), not in Python, so we only pull the
        rows we will actually show."""
        query = self._client.table("inspections").select("*")
        if site_name:
            query = query.eq("site_name", site_name)
        if media_type:
            query = query.eq("media_type", media_type)
        response = query.order("created_at", desc=True).limit(limit).execute()
        return list(response.data)

    def get_detections(self, inspection_id: str) -> list[dict[str, Any]]:
        response = (
            self._client.table("detections")
            .select("*")
            .eq("inspection_id", inspection_id)
            .execute()
        )
        return list(response.data)

    def fault_counts(
        self, site_name: str | None = None, inspection_ids: list[str] | None = None
    ) -> dict[str, int]:
        """Detection count per class - the data behind the History chart.

        Aggregation happens in Python (collections.Counter) rather than a
        Postgres group-by: PostgREST needs an RPC/stored procedure for real
        GROUP BY, and the row counts here are tiny. Keeping it client-side
        avoids a SQL function we would have to maintain.
        """
        query = self._client.table("detections").select("class_name,inspection_id")
        if inspection_ids is not None:
            if not inspection_ids:
                return {}
            query = query.in_("inspection_id", inspection_ids)
        response = query.execute()

        counts: dict[str, int] = {}
        for row in response.data:
            counts[row["class_name"]] = counts.get(row["class_name"], 0) + 1
        return counts

    def distinct_sites(self) -> list[str]:
        """Site names for the History filter dropdown."""
        response = self._client.table("inspections").select("site_name").execute()
        seen: list[str] = []
        for row in response.data:
            site = row["site_name"]
            if site not in seen:
                seen.append(site)
        return sorted(seen)


def detection_row(
    detection: Any,
) -> dict[str, Any]:
    """Map a pipeline Detection to the `detections` table schema.

    Kept separate from insert_detections so the mapping is unit-testable on
    its own (tests use a simple stand-in object with the same attributes).
    """
    return {
        "class_name": detection.class_name,
        "confidence": float(detection.confidence),
        "x1": float(detection.x1),
        "y1": float(detection.y1),
        "x2": float(detection.x2),
        "y2": float(detection.y2),
        "track_id": detection.track_id,
    }


def save_inspection(
    db: "InspectionsDB",
    site_name: str,
    source_filename: str,
    media_type: str,
    model_version: str,
    detections: list[Any],
) -> str:
    """Transactional-ish save used by the Upload screen: one parent row plus
    its detections. (PostgREST has no client-side transactions; if the
    detections insert fails the parent row remains - documented in TESTING.md
    as a known limitation.)"""
    inspection_id = db.insert_inspection(
        site_name=site_name,
        source_filename=source_filename,
        media_type=media_type,
        model_version=model_version,
        total_faults=len(detections),
    )
    db.insert_detections(inspection_id, [detection_row(d) for d in detections])
    return inspection_id
