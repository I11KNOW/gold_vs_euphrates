"""Unit tests for data loading, parsing, and fallback resilience."""

from __future__ import annotations

import json
import pandas as pd
import pytest

from src import data_loader


def test_fallback_euphrates_loads_through_2026():
    df = data_loader.load_fallback_euphrates_data()
    assert not df.empty
    assert list(df.columns) == ["datetime", "pass_date", "wse", "wse_u"]
    assert df["datetime"].min().year == 2016
    assert df["datetime"].max() >= pd.Timestamp("2026-10-15")
    assert len(df) >= 136
    assert df["wse"].isna().sum() == 0


def test_fallback_gold_loads_through_2026():
    df = data_loader.load_fallback_gold_prices()
    assert not df.empty
    assert list(df.columns) == ["Date", "Close"]
    assert df["Date"].min().year == 2016
    assert df["Date"].max() >= pd.Timestamp("2026-11-01")
    assert (df["Close"] > 0).all()
    assert df["Date"].is_monotonic_increasing


def test_parse_dahiti_station_html_extracts_levels_and_errors():
    pts = [[1463124153000.0, 10.9881], [1465456953000.0, 10.6945]]
    errs = [[1463124153000.0, 0.0304], [1465456953000.0, 1.1318]]
    html = (
        f'<script>var data_water_level = JSON.parse("{json.dumps(pts)}");'
        f'var data_water_level_error = JSON.parse("{json.dumps(errs)}");</script>'
    )
    df = data_loader.parse_dahiti_station_html(html)
    assert len(df) == 2
    assert df.loc[0, "wse"] == pytest.approx(10.988, abs=1e-3)
    assert df.loc[0, "wse_u"] == pytest.approx(0.030, abs=1e-3)
    assert df.loc[1, "wse_u"] == pytest.approx(1.132, abs=1e-3)


def test_load_euphrates_falls_back_gracefully_on_network_failure(monkeypatch):
    data_loader.clear_live_caches()

    def _boom(*_args, **_kwargs):
        raise ConnectionError("Simulated DAHITI outage")

    monkeypatch.setattr(data_loader, "_fetch_live_euphrates_cached", _boom)
    res = data_loader.load_euphrates_data()

    assert res.source_mode == "fallback"
    assert res.warning_message is not None
    assert "DAHITI live feed unavailable" in res.warning_message
    assert len(res.df) >= 136


def test_load_gold_falls_back_gracefully_on_rate_limit(monkeypatch):
    data_loader.clear_live_caches()

    def _boom(*_args, **_kwargs):
        raise RuntimeError("Simulated Yahoo 429 rate limit")

    monkeypatch.setattr(data_loader, "_fetch_live_gold_cached", _boom)
    res = data_loader.load_gold_prices()

    assert res.source_mode == "fallback"
    assert res.warning_message is not None
    assert "Live Gold feed unavailable" in res.warning_message
    assert len(res.df) > 2500
