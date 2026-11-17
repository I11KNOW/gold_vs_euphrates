"""Unit tests for frequency alignment, calendar-grid smoothing, and statistical integrity."""

from __future__ import annotations

from datetime import date
import pandas as pd
import pytest

from src.analysis import (
    align_datasets,
    apply_window_and_smoothing,
    build_annual_summary,
    classify_correlation,
    compute_rolling_correlation,
    compute_statistics,
)
from src.config import ALIGNMENT_MONTHLY, ALIGNMENT_PASS_DATE
from src.data_loader import load_fallback_euphrates_data, load_fallback_gold_prices


@pytest.fixture(scope="module")
def fallback_frames():
    return load_fallback_gold_prices(), load_fallback_euphrates_data()


def test_monthly_alignment_preserves_missing_months_and_clips_trailing_gold(fallback_frames):
    gold_df, eu_df = fallback_frames
    aligned, meta = align_datasets(gold_df, eu_df, alignment_mode=ALIGNMENT_MONTHLY)

    assert meta.missing_months_count == 2
    assert set(meta.missing_months) == {"2017-06", "2018-09"}
    assert meta.multi_pass_months_count >= 12
    # Must NOT include 2026-11 gold trading days since Euphrates ends 2026-10-15
    assert aligned["Date"].max() <= pd.Timestamp("2026-10-31")
    # Uniform calendar grid includes 2017-06 with Valid_Pair == False
    june_2017 = aligned[aligned["Date"] == pd.Timestamp("2017-06-30")]
    assert len(june_2017) == 1
    assert not bool(june_2017["Valid_Pair"].iloc[0])


def test_pass_date_alignment_matches_nearest_trading_day_without_dtype_error(fallback_frames):
    gold_df, eu_df = fallback_frames
    aligned, meta = align_datasets(gold_df, eu_df, alignment_mode=ALIGNMENT_PASS_DATE)

    assert len(aligned) == len(eu_df)
    assert meta.max_match_gap_days <= 3
    assert aligned["Valid_Pair"].all()


def test_uncertainty_filtering_drops_high_error_passes(fallback_frames):
    gold_df, eu_df = fallback_frames
    aligned, meta = align_datasets(
        gold_df,
        eu_df,
        alignment_mode=ALIGNMENT_PASS_DATE,
        max_uncertainty_m=0.5,
    )
    assert meta.dropped_uncertainty_rows > 0
    assert (aligned["WSE_Uncertainty_m"] <= 0.5).all()


def test_calendar_grid_rolling_does_not_bleed_across_missing_months():
    # Construct synthetic monthly series with a missing month in Feb 2020
    gold = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-15", "2020-02-15", "2020-03-15", "2020-04-15"]),
            "Close": [100.0, 200.0, 300.0, 400.0],
        }
    )
    eu = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-15", "2020-03-15", "2020-04-15"]),
            "pass_date": pd.to_datetime(["2020-01-15", "2020-03-15", "2020-04-15"]),
            "wse": [10.0, 12.0, 14.0],
            "wse_u": [0.05, 0.05, 0.05],
        }
    )
    aligned, _ = align_datasets(gold, eu, alignment_mode=ALIGNMENT_MONTHLY)
    smoothed = apply_window_and_smoothing(
        aligned,
        start_date=date(2020, 1, 1),
        end_date=date(2020, 4, 30),
        smoothing_window=2,
        alignment_mode=ALIGNMENT_MONTHLY,
    )
    # March 2020 2-month calendar window covers [Feb (NaN), Mar (12.0)], so mean MUST be 12.0, NOT (10+12)/2 = 11.0!
    mar = smoothed[smoothed["Date"] == pd.Timestamp("2020-03-31")].iloc[0]
    assert mar["Euphrates_Plot"] == pytest.approx(12.0)


def test_positive_correlation_window_is_classified_accurately(fallback_frames):
    gold_df, eu_df = fallback_frames
    aligned, _ = align_datasets(gold_df, eu_df, alignment_mode=ALIGNMENT_MONTHLY)
    windowed = apply_window_and_smoothing(
        aligned,
        start_date=date(2018, 1, 1),
        end_date=date(2020, 8, 31),
        smoothing_window=1,
        alignment_mode=ALIGNMENT_MONTHLY,
    )
    summary = compute_statistics(windowed, alignment_mode=ALIGNMENT_MONTHLY)
    assert summary.is_valid
    assert summary.pearson_r > 0.5
    assert summary.correlation_label == "Strong Positive"


def test_small_sample_and_constant_series_do_not_crash(fallback_frames):
    gold_df, eu_df = fallback_frames
    aligned, _ = align_datasets(gold_df, eu_df, alignment_mode=ALIGNMENT_MONTHLY)

    single = apply_window_and_smoothing(
        aligned,
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 31),
        smoothing_window=1,
        alignment_mode=ALIGNMENT_MONTHLY,
    )
    s1 = compute_statistics(single, alignment_mode=ALIGNMENT_MONTHLY)
    assert not s1.is_valid

    const_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "Year": [2020, 2020, 2020],
            "Gold_Price": [1500.0, 1550.0, 1600.0],
            "Euphrates_Level_m": [11.0, 11.0, 11.0],
        }
    )
    s_const = compute_statistics(const_df, alignment_mode=ALIGNMENT_MONTHLY)
    assert not s_const.is_valid
    assert "zero variance" in (s_const.invalid_reason or "")


def test_rolling_correlation_and_annual_summary(fallback_frames):
    gold_df, eu_df = fallback_frames
    aligned, _ = align_datasets(gold_df, eu_df, alignment_mode=ALIGNMENT_MONTHLY)
    valid = apply_window_and_smoothing(
        aligned,
        start_date=date(2016, 5, 1),
        end_date=date(2026, 10, 31),
        smoothing_window=3,
        alignment_mode=ALIGNMENT_MONTHLY,
    )
    roll = compute_rolling_correlation(valid, window=12)
    assert not roll.empty
    assert roll["Rolling_Pearson_r"].between(-1.0, 1.0).all()

    annual = build_annual_summary(valid)
    assert 2016 in annual.index and 2026 in annual.index
    assert "Observations" in annual.columns
    assert "Mean_Uncertainty_m" in annual.columns
    assert classify_correlation(-0.6) == "Strong Inverse"
    assert classify_correlation(0.05) == "Weak / Negligible"
