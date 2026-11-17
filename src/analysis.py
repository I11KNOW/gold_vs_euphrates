"""Time-series alignment, uncertainty filtering, smoothing, and statistical analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    ALIGNMENT_MONTHLY,
    ALIGNMENT_PASS_DATE,
    PASS_MATCH_TOLERANCE_DAYS,
    ROLLING_CORR_WINDOW,
)


@dataclass(frozen=True)
class AlignmentMetadata:
    """Diagnostics describing how the two heterogeneous series were aligned."""

    mode: str
    raw_euphrates_rows: int
    filtered_euphrates_rows: int
    dropped_uncertainty_rows: int
    aligned_valid_rows: int
    calendar_grid_months: int
    missing_months_count: int
    missing_months: tuple[str, ...]
    multi_pass_months_count: int
    max_match_gap_days: int


@dataclass(frozen=True)
class StatsSummary:
    """Level-space and first-differenced statistical summary."""

    is_valid: bool
    n_obs: int
    invalid_reason: Optional[str] = None
    pearson_r: float = float("nan")
    pearson_p: float = float("nan")
    spearman_rho: float = float("nan")
    spearman_p: float = float("nan")
    ols_slope: float = float("nan")
    ols_intercept: float = float("nan")
    r_squared: float = float("nan")
    ols_std_err: float = float("nan")
    diff_n: int = 0
    diff_pearson_r: float = float("nan")
    diff_pearson_p: float = float("nan")
    correlation_label: str = "Insufficient Data"


def filter_euphrates_by_uncertainty(
    euphrates_df: pd.DataFrame,
    max_uncertainty_m: Optional[float],
) -> tuple[pd.DataFrame, int]:
    """Filter satellite altimetry readings by maximum uncertainty threshold (wse_u)."""
    if max_uncertainty_m is None:
        return euphrates_df.copy(), 0

    mask = euphrates_df["wse_u"].isna() | (euphrates_df["wse_u"] <= max_uncertainty_m)
    filtered = euphrates_df.loc[mask].copy().reset_index(drop=True)
    dropped = int(len(euphrates_df) - len(filtered))
    return filtered, dropped


def align_datasets(
    gold_df: pd.DataFrame,
    euphrates_df: pd.DataFrame,
    alignment_mode: str = ALIGNMENT_MONTHLY,
    max_uncertainty_m: Optional[float] = None,
) -> tuple[pd.DataFrame, AlignmentMetadata]:
    """Align daily Gold prices and periodic Euphrates water levels without silent bias."""
    raw_count = len(euphrates_df)
    eu_filtered, dropped_count = filter_euphrates_by_uncertainty(euphrates_df, max_uncertainty_m)

    if eu_filtered.empty or gold_df.empty:
        empty_cols = [
            "Date",
            "Year",
            "Gold_Price",
            "Euphrates_Level_m",
            "WSE_Uncertainty_m",
            "Pass_Count",
            "Gold_Trading_Days",
            "Alignment_Lag_Days",
            "Valid_Pair",
        ]
        meta = AlignmentMetadata(
            mode=alignment_mode,
            raw_euphrates_rows=raw_count,
            filtered_euphrates_rows=len(eu_filtered),
            dropped_uncertainty_rows=dropped_count,
            aligned_valid_rows=0,
            calendar_grid_months=0,
            missing_months_count=0,
            missing_months=(),
            multi_pass_months_count=0,
            max_match_gap_days=0,
        )
        return pd.DataFrame(columns=empty_cols), meta

    if alignment_mode == ALIGNMENT_MONTHLY:
        eu_monthly = eu_filtered.set_index("datetime").resample("ME").agg(
            Euphrates_Level_m=("wse", "mean"),
            WSE_Uncertainty_m=("wse_u", "mean"),
            Pass_Count=("wse", "count"),
        )
        # Clip Gold before resampling so trailing unpaired days (e.g. partial current month)
        # cannot bleed past the latest available river observation month.
        latest_eu_month_end = eu_monthly.index.max()
        gold_clipped = gold_df[gold_df["Date"] <= latest_eu_month_end]

        gold_monthly = gold_clipped.set_index("Date").resample("ME").agg(
            Gold_Price=("Close", "mean"),
            Gold_Trading_Days=("Close", "count"),
        )

        if gold_monthly.empty or eu_monthly.empty:
            return align_datasets(gold_df.iloc[0:0], eu_filtered, alignment_mode, None)

        start_me = max(eu_monthly.index.min(), gold_monthly.index.min())
        end_me = min(eu_monthly.index.max(), gold_monthly.index.max())
        full_grid = pd.date_range(start_me, end_me, freq="ME").astype("datetime64[ns]")

        aligned = pd.concat([gold_monthly, eu_monthly], axis=1).reindex(full_grid)
        aligned.index.name = "Date"
        aligned = aligned.reset_index()
        aligned["Date"] = aligned["Date"].astype("datetime64[ns]")
        aligned["Year"] = aligned["Date"].dt.year
        aligned["Pass_Count"] = aligned["Pass_Count"].fillna(0).astype(int)
        aligned["Gold_Trading_Days"] = aligned["Gold_Trading_Days"].fillna(0).astype(int)
        aligned["Alignment_Lag_Days"] = 0
        aligned["Valid_Pair"] = (
            aligned["Gold_Price"].notna() & aligned["Euphrates_Level_m"].notna()
        )

        missing_mask = ~aligned["Valid_Pair"]
        missing_labels = tuple(aligned.loc[missing_mask, "Date"].dt.strftime("%Y-%m").tolist())
        multi_pass_count = int((aligned["Pass_Count"] > 1).sum())

        meta = AlignmentMetadata(
            mode=alignment_mode,
            raw_euphrates_rows=raw_count,
            filtered_euphrates_rows=len(eu_filtered),
            dropped_uncertainty_rows=dropped_count,
            aligned_valid_rows=int(aligned["Valid_Pair"].sum()),
            calendar_grid_months=len(aligned),
            missing_months_count=len(missing_labels),
            missing_months=missing_labels,
            multi_pass_months_count=multi_pass_count,
            max_match_gap_days=0,
        )
        return aligned, meta

    if alignment_mode == ALIGNMENT_PASS_DATE:
        eu_sorted = eu_filtered.sort_values("pass_date").copy()
        gold_sorted = (
            gold_df[["Date", "Close"]]
            .rename(columns={"Date": "Gold_Date", "Close": "Gold_Price"})
            .sort_values("Gold_Date")
            .copy()
        )
        eu_sorted["pass_date"] = eu_sorted["pass_date"].astype("datetime64[ns]")
        gold_sorted["Gold_Date"] = gold_sorted["Gold_Date"].astype("datetime64[ns]")

        matched = pd.merge_asof(
            eu_sorted,
            gold_sorted,
            left_on="pass_date",
            right_on="Gold_Date",
            direction="nearest",
            tolerance=pd.Timedelta(days=PASS_MATCH_TOLERANCE_DAYS),
        ).dropna(subset=["Gold_Price", "wse"]).reset_index(drop=True)

        aligned = pd.DataFrame(
            {
                "Date": matched["pass_date"].astype("datetime64[ns]"),
                "Year": matched["pass_date"].dt.year.astype(int),
                "Gold_Price": matched["Gold_Price"].astype(float),
                "Euphrates_Level_m": matched["wse"].astype(float),
                "WSE_Uncertainty_m": matched["wse_u"].astype(float),
                "Pass_Count": 1,
                "Gold_Trading_Days": 1,
                "Alignment_Lag_Days": (matched["pass_date"] - matched["Gold_Date"])
                .abs()
                .dt.days.astype(int),
                "Valid_Pair": True,
            }
        )
        max_gap = int(aligned["Alignment_Lag_Days"].max()) if not aligned.empty else 0

        meta = AlignmentMetadata(
            mode=alignment_mode,
            raw_euphrates_rows=raw_count,
            filtered_euphrates_rows=len(eu_filtered),
            dropped_uncertainty_rows=dropped_count,
            aligned_valid_rows=len(aligned),
            calendar_grid_months=0,
            missing_months_count=0,
            missing_months=(),
            multi_pass_months_count=0,
            max_match_gap_days=max_gap,
        )
        return aligned, meta

    raise ValueError(f"Unsupported alignment mode: {alignment_mode}")


def apply_window_and_smoothing(
    aligned_df: pd.DataFrame,
    start_date: date,
    end_date: date,
    smoothing_window: int,
    alignment_mode: str,
) -> pd.DataFrame:
    """Slice to the date window, apply gap-aware rolling smoothing, and normalize Base-100."""
    if aligned_df.empty:
        return aligned_df.copy()

    mask = (aligned_df["Date"].dt.date >= start_date) & (aligned_df["Date"].dt.date <= end_date)
    windowed = aligned_df.loc[mask].copy().reset_index(drop=True)
    if windowed.empty:
        return windowed

    # Compute rolling averages on the full calendar grid in monthly mode so missing months
    # remain NaN gaps instead of silently compressing distant calendar months together.
    if smoothing_window > 1:
        windowed["Gold_Plot"] = (
            windowed["Gold_Price"].rolling(smoothing_window, min_periods=1).mean()
        )
        windowed["Euphrates_Plot"] = (
            windowed["Euphrates_Level_m"].rolling(smoothing_window, min_periods=1).mean()
        )
    else:
        windowed["Gold_Plot"] = windowed["Gold_Price"]
        windowed["Euphrates_Plot"] = windowed["Euphrates_Level_m"]

    valid = windowed.loc[windowed["Valid_Pair"]].copy().reset_index(drop=True)
    if valid.empty:
        return valid

    gold_base = valid["Gold_Price"].iloc[0]
    water_base = valid["Euphrates_Level_m"].iloc[0]
    valid["Gold_Norm_Dynamic"] = (
        (valid["Gold_Price"] / gold_base) * 100.0 if gold_base else np.nan
    )
    valid["Euphrates_Norm_Dynamic"] = (
        (valid["Euphrates_Level_m"] / water_base) * 100.0 if water_base else np.nan
    )
    valid["Gold_Norm_Plot"] = (
        (valid["Gold_Plot"] / gold_base) * 100.0 if gold_base else np.nan
    )
    valid["Euphrates_Norm_Plot"] = (
        (valid["Euphrates_Plot"] / water_base) * 100.0 if water_base else np.nan
    )
    return valid


def classify_correlation(pearson_r: float) -> str:
    """Return an accurate directional strength label for a Pearson correlation value."""
    if np.isnan(pearson_r):
        return "Insufficient Data"
    if pearson_r <= -0.5:
        return "Strong Inverse"
    if pearson_r <= -0.2:
        return "Moderate Inverse"
    if pearson_r < 0.2:
        return "Weak / Negligible"
    if pearson_r < 0.5:
        return "Moderate Positive"
    return "Strong Positive"


def compute_statistics(valid_df: pd.DataFrame, alignment_mode: str) -> StatsSummary:
    """Compute level-space and consecutive first-differenced statistical metrics safely."""
    n_obs = len(valid_df)
    if n_obs < 3:
        return StatsSummary(
            is_valid=False,
            n_obs=n_obs,
            invalid_reason="At least 3 paired observations are required for correlation and OLS regression.",
        )

    x = valid_df["Euphrates_Level_m"].to_numpy(dtype=float)
    y = valid_df["Gold_Price"].to_numpy(dtype=float)

    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return StatsSummary(
            is_valid=False,
            n_obs=n_obs,
            invalid_reason="Selected window has zero variance in one of the series.",
        )

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_rho, spearman_p = stats.spearmanr(x, y)
    reg = stats.linregress(x, y)

    # First-differenced correlation checks whether period-to-period changes actually co-move
    # once long-term secular trends (non-stationarity) are removed.
    if alignment_mode == ALIGNMENT_MONTHLY:
        ym = valid_df["Date"].dt.year * 12 + valid_df["Date"].dt.month
        consecutive = ym.diff() == 1
    else:
        consecutive = valid_df["Date"].diff().dt.days <= 35

    diff_df = pd.DataFrame(
        {
            "d_water": valid_df["Euphrates_Level_m"].diff(),
            "d_gold": valid_df["Gold_Price"].diff(),
        }
    ).loc[consecutive].dropna()

    diff_n = len(diff_df)
    if diff_n >= 3 and diff_df["d_water"].std() > 0 and diff_df["d_gold"].std() > 0:
        diff_r, diff_p = stats.pearsonr(diff_df["d_water"], diff_df["d_gold"])
    else:
        diff_r, diff_p = float("nan"), float("nan")

    return StatsSummary(
        is_valid=True,
        n_obs=n_obs,
        pearson_r=float(pearson_r),
        pearson_p=float(pearson_p),
        spearman_rho=float(spearman_rho),
        spearman_p=float(spearman_p),
        ols_slope=float(reg.slope),
        ols_intercept=float(reg.intercept),
        r_squared=float(reg.rvalue ** 2),
        ols_std_err=float(reg.stderr),
        diff_n=diff_n,
        diff_pearson_r=float(diff_r),
        diff_pearson_p=float(diff_p),
        correlation_label=classify_correlation(float(pearson_r)),
    )


def compute_rolling_correlation(
    valid_df: pd.DataFrame,
    window: int = ROLLING_CORR_WINDOW,
) -> pd.DataFrame:
    """Compute a rolling Pearson correlation series to reveal time-varying regime shifts."""
    if len(valid_df) < window:
        return pd.DataFrame(columns=["Date", "Rolling_Pearson_r"])

    rolling_r = valid_df["Gold_Price"].rolling(window=window).corr(
        valid_df["Euphrates_Level_m"]
    )
    out = pd.DataFrame({"Date": valid_df["Date"], "Rolling_Pearson_r": rolling_r})
    return out.dropna(subset=["Rolling_Pearson_r"]).reset_index(drop=True)


def build_annual_summary(valid_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate yearly summary metrics including observation counts and uncertainty."""
    if valid_df.empty:
        return pd.DataFrame()

    summary = (
        valid_df.groupby("Year")
        .agg(
            Observations=("Date", "count"),
            Gold_Avg=("Gold_Price", "mean"),
            Gold_Min=("Gold_Price", "min"),
            Gold_Max=("Gold_Price", "max"),
            Water_Level_Avg=("Euphrates_Level_m", "mean"),
            Water_Level_Min=("Euphrates_Level_m", "min"),
            Water_Level_Max=("Euphrates_Level_m", "max"),
            Mean_Uncertainty_m=("WSE_Uncertainty_m", "mean"),
        )
        .round(2)
    )
    return summary
