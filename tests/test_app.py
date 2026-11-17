"""End-to-end Streamlit AppTest suite exercising every control, tab, and fallback path."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src import data_loader
from src.config import ALIGNMENT_PASS_DATE

APP_PATH = str(Path(__file__).resolve().parent.parent / "dashboard.py")


def _force_fallback(monkeypatch):
    """Force deterministic local fallback mode so tests run hermetically offline."""
    data_loader.clear_live_caches()
    monkeypatch.setattr(
        data_loader,
        "_fetch_live_euphrates_cached",
        lambda *_a, **_kw: (_ for _ in ()).throw(ConnectionError("offline test")),
    )
    monkeypatch.setattr(
        data_loader,
        "_fetch_live_gold_cached",
        lambda *_a, **_kw: (_ for _ in ()).throw(ConnectionError("offline test")),
    )


def test_app_boots_cleanly_in_offline_fallback_mode(monkeypatch):
    _force_fallback(monkeypatch)
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    assert not at.exception
    assert not at.error
    assert len(at.warning) == 1  # Fallback banner shown clearly
    assert len(at.metric) == 4
    assert len(at.tabs) == 3
    assert len(at.dataframe) >= 1


def test_all_interactive_sidebar_controls(monkeypatch):
    _force_fallback(monkeypatch)
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    # 1. Switch alignment mode to Satellite Pass Date
    at.sidebar.radio(key="alignment_mode").set_value(ALIGNMENT_PASS_DATE).run()
    assert not at.exception
    assert "Passes" in at.metric[0].value

    # 2. Tighten altimetry uncertainty filter to <= 0.5m
    at.sidebar.slider(key="max_uncertainty").set_value(0.5).run()
    assert not at.exception
    assert int(at.metric[0].value.split()[0]) < 136

    # 3. Toggle Normalized Base-100% checkbox ON
    at.sidebar.checkbox(key="show_normalized").set_value(True).run()
    assert not at.exception
    assert at.sidebar.checkbox(key="show_uncertainty").disabled

    # 4. Toggle Normalized OFF and Uncertainty Band ON
    at.sidebar.checkbox(key="show_normalized").set_value(False).run()
    at.sidebar.checkbox(key="show_uncertainty").set_value(True).run()
    assert not at.exception

    # 5. Adjust smoothing slider
    at.sidebar.slider(key="smoothing_window").set_value(6).run()
    assert not at.exception

    # 6. Select positive-correlation sub-window (2018-2020)
    at.sidebar.date_input(key="date_range").set_value(
        (date(2018, 1, 1), date(2020, 8, 31))
    ).run()
    assert not at.exception
    assert at.metric[3].delta == "Strong Positive"

    # 7. Narrow range to 1 month (insufficient for correlation -> graceful info, no crash)
    at.sidebar.date_input(key="date_range").set_value(
        (date(2020, 1, 1), date(2020, 1, 31))
    ).run()
    assert not at.exception
    assert at.metric[3].value == "N/A"

    # 8. Click Refresh Live Data button cleanly
    at.sidebar.button(key="refresh_data_btn").click().run()
    assert not at.exception
    assert not at.error

    # 9. Switch back to Monthly Mean and reset date range to full span
    from src.config import ALIGNMENT_MONTHLY

    at.sidebar.radio(key="alignment_mode").set_value(ALIGNMENT_MONTHLY).run()
    at.sidebar.slider(key="max_uncertainty").set_value(2.5).run()
    at.sidebar.date_input(key="date_range").set_value(
        (date(2016, 5, 31), date(2026, 10, 31))
    ).run()
    assert not at.exception
    assert len(at.metric) == 4
    assert "Months" in at.metric[0].value
