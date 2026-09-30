"""Smoke test: the real Streamlit app boots via streamlit.testing.v1.AppTest.

This runs the ACTUAL app.py in a headless runtime - no browser, no model
download, no secrets. It catches broken imports, syntax errors and crashes in
sidebar/module top-level code.
"""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

# AppTest resolves relative paths against THIS file's location, not the cwd,
# so build an explicit path to the project root's app.py.
APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_boots_without_model_or_secrets():
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()

    assert not at.exception, f"app raised: {at.exception}"
    # Sidebar navigation radio exists (find by LABEL, not index: when a model
    # file exists, the Upload screen adds its own 'Media type' radio).
    nav = [r for r in at.radio if r.label == "Screen"]
    assert nav, "sidebar navigation radio missing"
    assert nav[0].value in ("Upload", "History")


def test_app_switches_to_history_screen():
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    nav = [r for r in at.radio if r.label == "Screen"][0]
    nav.set_value("History").run()
    assert not at.exception
