"""Resilient data ingestion, caching, and local fallback loaders."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from src.config import (
    DAHITI_API_V2_URL,
    DAHITI_STATION_ID,
    DAHITI_STATION_PAGE_URL,
    EUPHRATES_CACHE_TTL_SECONDS,
    EUPHRATES_FALLBACK_PATH,
    EUPHRATES_LEGACY_PATH,
    GOLD_CACHE_TTL_SECONDS,
    GOLD_FALLBACK_PATH,
    GOLD_START_DATE,
    GOLD_TICKER,
    HTTP_TIMEOUT_SECONDS,
    YAHOO_CHART_V8_URL,
)

USER_AGENT = "Mozilla/5.0 (compatible; GoldVsEuphratesDashboard/2.0)"


@dataclass(frozen=True)
class DatasetResult:
    """Container for a loaded dataset and its provenance metadata."""

    df: pd.DataFrame
    source_mode: str       # "live_api", "live_public", or "fallback"
    source_detail: str
    latest_date: pd.Timestamp
    warning_message: Optional[str] = None


def _get_dahiti_api_key() -> Optional[str]:
    """Read an optional DAHITI API v2 key from env vars or Streamlit secrets."""
    env_key = os.environ.get("DAHITI_API_KEY", "").strip()
    if env_key:
        return env_key

    try:
        secret_val = st.secrets.get("DAHITI_API_KEY", "")
        if isinstance(secret_val, str) and secret_val.strip():
            return secret_val.strip()
    except Exception:
        # Local runs without .streamlit/secrets.toml raise when accessing st.secrets
        return None
    return None


def _normalize_euphrates_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardize Euphrates water level observations."""
    required = {"datetime", "wse"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Euphrates dataset missing required columns: {missing}")

    out = df.copy()
    out["datetime"] = (
        pd.to_datetime(out["datetime"], errors="coerce")
        .dt.tz_localize(None)
        .astype("datetime64[ns]")
    )
    out["wse"] = pd.to_numeric(out["wse"], errors="coerce")
    if "wse_u" in out.columns:
        out["wse_u"] = pd.to_numeric(out["wse_u"], errors="coerce")
    else:
        out["wse_u"] = np.nan

    out = (
        out.dropna(subset=["datetime", "wse"])
        .sort_values("datetime")
        .drop_duplicates(subset=["datetime"], keep="last")
        .reset_index(drop=True)
    )
    if out.empty:
        raise ValueError("Euphrates dataset contains zero valid rows after cleaning.")

    out["pass_date"] = out["datetime"].dt.normalize().astype("datetime64[ns]")
    return out[["datetime", "pass_date", "wse", "wse_u"]]


def _normalize_gold_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardize daily Gold futures closing prices."""
    if df.empty:
        raise ValueError("Gold dataset is empty.")

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        if "Close" in out.columns.get_level_values(0):
            close_obj = out["Close"]
            close_series = close_obj.iloc[:, 0] if isinstance(close_obj, pd.DataFrame) else close_obj
            out = close_series.to_frame(name="Close")
        else:
            out.columns = out.columns.get_level_values(0)

    if "Date" not in out.columns:
        out = out.reset_index()

    date_col = "Date" if "Date" in out.columns else out.columns[0]
    if "Close" not in out.columns:
        raise ValueError(f"Gold dataset missing 'Close' column (found: {list(out.columns)})")

    out = out[[date_col, "Close"]].rename(columns={date_col: "Date"})
    out["Date"] = (
        pd.to_datetime(out["Date"], errors="coerce")
        .dt.tz_localize(None)
        .dt.normalize()
        .astype("datetime64[ns]")
    )
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna(subset=["Date", "Close"])
    out = out[out["Close"] > 0]
    out = (
        out.sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )
    if len(out) < 50:
        raise ValueError(f"Gold dataset unexpectedly short ({len(out)} rows).")

    return out[["Date", "Close"]]


def parse_dahiti_station_html(html_text: str) -> pd.DataFrame:
    """Extract water level and uncertainty series from DAHITI's public station page."""
    level_match = re.search(
        r'var\s+data_water_level\s*=\s*JSON\.parse\((["\'])(.*?)\1\)',
        html_text,
        flags=re.DOTALL,
    )
    if not level_match:
        raise ValueError("Could not locate 'data_water_level' in DAHITI station HTML.")

    level_points = json.loads(level_match.group(2))
    if not isinstance(level_points, list) or not level_points:
        raise ValueError("Parsed DAHITI 'data_water_level' is empty.")

    error_map: dict[int, float] = {}
    error_match = re.search(
        r'var\s+data_water_level_error\s*=\s*JSON\.parse\((["\'])(.*?)\1\)',
        html_text,
        flags=re.DOTALL,
    )
    if error_match:
        error_points = json.loads(error_match.group(2))
        for ts_ms, err_val in error_points:
            error_map[int(ts_ms)] = float(err_val)

    rows = []
    for ts_ms, wse_val in level_points:
        ts_int = int(ts_ms)
        dt = pd.to_datetime(ts_int, unit="ms", utc=True).tz_localize(None)
        rows.append(
            {
                "datetime": dt,
                "wse": round(float(wse_val), 3),
                "wse_u": round(error_map[ts_int], 3) if ts_int in error_map else np.nan,
            }
        )

    return _normalize_euphrates_frame(pd.DataFrame(rows))


def _fetch_dahiti_api_v2(api_key: str) -> pd.DataFrame:
    """Fetch water level series via the official DAHITI API v2 endpoint."""
    response = requests.post(
        DAHITI_API_V2_URL,
        data={
            "api_key": api_key,
            "dahiti_id": DAHITI_STATION_ID,
            "format": "json",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != 200 or not payload.get("data"):
        raise ValueError(f"DAHITI API v2 returned unexpected payload: {payload.get('message')}")

    df = pd.DataFrame(payload["data"]).rename(columns={"date": "datetime"})
    return _normalize_euphrates_frame(df)


def _fetch_dahiti_public_station() -> pd.DataFrame:
    """Fetch water level series from DAHITI's public station altimetry page."""
    response = requests.get(
        DAHITI_STATION_PAGE_URL,
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return parse_dahiti_station_html(response.text)


@st.cache_data(ttl=EUPHRATES_CACHE_TTL_SECONDS, show_spinner=False)
def _fetch_live_euphrates_cached(api_key: str) -> tuple[pd.DataFrame, str, str]:
    """Cached live Euphrates loader; raises on failure so errors are not cached."""
    if api_key:
        try:
            df = _fetch_dahiti_api_v2(api_key)
            return df, "live_api", f"DAHITI API v2 (Station {DAHITI_STATION_ID})"
        except Exception:
            pass

    df = _fetch_dahiti_public_station()
    return df, "live_public", f"DAHITI Station {DAHITI_STATION_ID} Live Feed"


def _resolve_euphrates_fallback_path() -> Path:
    """Return the first existing local Euphrates CSV fallback path."""
    for candidate in (EUPHRATES_FALLBACK_PATH, EUPHRATES_LEGACY_PATH):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Local Euphrates fallback CSV not found at {EUPHRATES_FALLBACK_PATH}."
    )


@st.cache_data(show_spinner=False)
def load_fallback_euphrates_data() -> pd.DataFrame:
    """Load and normalize the bundled Euphrates CSV fallback dataset."""
    csv_path = _resolve_euphrates_fallback_path()
    df = pd.read_csv(csv_path, sep=";")
    return _normalize_euphrates_frame(df)


def load_euphrates_data() -> DatasetResult:
    """Load Euphrates water level readings with automatic local CSV fallback."""
    fallback_df = load_fallback_euphrates_data()
    api_key = _get_dahiti_api_key() or ""

    try:
        live_df, mode, detail = _fetch_live_euphrates_cached(api_key)
        if (
            live_df["datetime"].max() >= fallback_df["datetime"].max()
            and len(live_df) >= len(fallback_df) * 0.9
        ):
            return DatasetResult(
                df=live_df,
                source_mode=mode,
                source_detail=detail,
                latest_date=live_df["datetime"].max(),
            )
    except Exception as exc:
        return DatasetResult(
            df=fallback_df,
            source_mode="fallback",
            source_detail="Bundled DAHITI Snapshot (data/euphrates_data.csv)",
            latest_date=fallback_df["datetime"].max(),
            warning_message=f"DAHITI live feed unavailable ({type(exc).__name__}); using bundled repository dataset.",
        )

    return DatasetResult(
        df=fallback_df,
        source_mode="fallback",
        source_detail="Bundled DAHITI Snapshot (data/euphrates_data.csv)",
        latest_date=fallback_df["datetime"].max(),
        warning_message="DAHITI live feed returned fewer records than the local snapshot; using bundled dataset.",
    )


def _download_gold_via_yfinance(start_date_str: str) -> pd.DataFrame:
    """Fetch Gold futures via yfinance with a non-blocking thread timeout guard."""
    def _run_download() -> pd.DataFrame:
        return yf.download(
            GOLD_TICKER,
            start=start_date_str,
            interval="1d",
            auto_adjust=True,
            multi_level_index=False,
            progress=False,
            timeout=HTTP_TIMEOUT_SECONDS,
        )

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_run_download)
    try:
        raw = future.result(timeout=HTTP_TIMEOUT_SECONDS + 3)
    except FuturesTimeoutError as exc:
        raise TimeoutError(
            f"yfinance request timed out after {HTTP_TIMEOUT_SECONDS + 3}s"
        ) from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return _normalize_gold_frame(raw)


def _download_gold_via_yahoo_chart(start_date_str: str) -> pd.DataFrame:
    """Fetch daily Gold futures directly from Yahoo Finance chart JSON endpoint."""
    period1 = int(pd.Timestamp(start_date_str, tz="UTC").timestamp())
    period2 = int(time.time()) + 86400
    response = requests.get(
        YAHOO_CHART_V8_URL,
        params={"period1": period1, "period2": period2, "interval": "1d"},
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()

    results = payload.get("chart", {}).get("result") or []
    if not results:
        raise ValueError("Yahoo Finance chart response contained no result payload.")

    result = results[0]
    timestamps = result.get("timestamp") or []
    quotes = result.get("indicators", {}).get("quote") or []
    if not timestamps or not quotes or "close" not in quotes[0]:
        raise ValueError("Yahoo Finance chart response missing timestamps or close prices.")

    dates = (
        pd.to_datetime(timestamps, unit="s", utc=True)
        .tz_convert("America/New_York")
        .normalize()
        .tz_localize(None)
    )
    df = pd.DataFrame({"Date": dates, "Close": quotes[0]["close"]})
    return _normalize_gold_frame(df)


@st.cache_data(ttl=GOLD_CACHE_TTL_SECONDS, show_spinner=False)
def _fetch_live_gold_cached(start_date_str: str) -> tuple[pd.DataFrame, str]:
    """Cached live Gold loader; raises if all live providers fail."""
    errors: list[str] = []
    for label, fetcher in (
        ("Yahoo Finance (yfinance)", _download_gold_via_yfinance),
        ("Yahoo Finance Chart v8 API", _download_gold_via_yahoo_chart),
    ):
        try:
            df = fetcher(start_date_str)
            return df, label
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}")

    raise RuntimeError("; ".join(errors))


@st.cache_data(show_spinner=False)
def load_fallback_gold_prices() -> pd.DataFrame:
    """Load and normalize the bundled Gold futures CSV fallback dataset."""
    if not GOLD_FALLBACK_PATH.exists():
        raise FileNotFoundError(
            f"Local Gold fallback CSV not found at {GOLD_FALLBACK_PATH}."
        )
    df = pd.read_csv(GOLD_FALLBACK_PATH)
    return _normalize_gold_frame(df)


def load_gold_prices(start_date_str: str = GOLD_START_DATE) -> DatasetResult:
    """Load Gold futures prices with multi-source live fetch and local CSV fallback."""
    fallback_df = load_fallback_gold_prices()

    try:
        live_df, detail = _fetch_live_gold_cached(start_date_str)
        if len(live_df) >= len(fallback_df) * 0.9:
            return DatasetResult(
                df=live_df,
                source_mode="live_api",
                source_detail=detail,
                latest_date=live_df["Date"].max(),
            )
    except Exception as exc:
        return DatasetResult(
            df=fallback_df,
            source_mode="fallback",
            source_detail="Bundled Gold Snapshot (data/gold_prices_fallback.csv)",
            latest_date=fallback_df["Date"].max(),
            warning_message=f"Live Gold feed unavailable ({exc}); using bundled repository snapshot.",
        )

    return DatasetResult(
        df=fallback_df,
        source_mode="fallback",
        source_detail="Bundled Gold Snapshot (data/gold_prices_fallback.csv)",
        latest_date=fallback_df["Date"].max(),
        warning_message="Live Gold feed returned incomplete history; using bundled repository snapshot.",
    )


def clear_live_caches() -> None:
    """Invalidate cached live data fetchers so the next run re-fetches from upstream."""
    for cached_fn in (_fetch_live_euphrates_cached, _fetch_live_gold_cached):
        if hasattr(cached_fn, "clear"):
            cached_fn.clear()
