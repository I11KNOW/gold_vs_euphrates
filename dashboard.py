"""
Streamlit Dashboard: Gold Price vs. Euphrates River Water Level (2016 - 2026)
=============================================================================
Interactive EDA dashboard comparing COMEX Gold Futures (GC=F) with DAHITI
satellite radar altimetry measurements for Euphrates River Station 41518 (Iraq).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analysis import (
    align_datasets,
    apply_window_and_smoothing,
    build_annual_summary,
    compute_rolling_correlation,
    compute_statistics,
)
from src.charts import (
    build_rolling_correlation_chart,
    build_scatter_regression_chart,
    build_time_series_chart,
)
from src.config import (
    ALIGNMENT_MODES,
    ALIGNMENT_MONTHLY,
    DAHITI_SATELLITE_MISSION,
    DAHITI_STATION_ID,
    DAHITI_STATION_LAT,
    DAHITI_STATION_LON,
    ROLLING_CORR_WINDOW,
)
from src.data_loader import clear_live_caches, load_euphrates_data, load_gold_prices

# ─── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold Price vs Euphrates Water Level (2016-2026)",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Header Section ─────────────────────────────────────────────────────────
st.title("🥇 Gold Price vs 🌊 Euphrates River Water Level")
st.markdown(
    f"""
Interactive EDA dashboard comparing COMEX Gold Futures (`GC=F`) against satellite altimetry
measurements of the Euphrates River (**DAHITI Station {DAHITI_STATION_ID}**, Iraq —
`{DAHITI_STATION_LAT:.4f}°N, {DAHITI_STATION_LON:.4f}°E`) from 2016 to 2026.

---

**قال رسول الله -صلى الله عليه وسلم-: (يوشك الفرات أن يحسر عن كنز من ذهب، فمن حضره فلا يأخذ منه شيئا) متفق عليه**
"""
)
st.divider()

# ─── Data Ingestion (Live with Automatic Local Fallback) ────────────────────
try:
    euphrates_res = load_euphrates_data()
    gold_res = load_gold_prices()
except Exception as exc:
    st.error(f"Critical error loading dashboard datasets: {exc}")
    st.stop()

fallback_warnings = [
    msg
    for msg in (euphrates_res.warning_message, gold_res.warning_message)
    if msg
]
if fallback_warnings:
    st.warning("⚠️ " + " | ".join(fallback_warnings))

# ─── Sidebar Controls ───────────────────────────────────────────────────────
st.sidebar.header("⚙️ Controls & Alignment")

alignment_mode = st.sidebar.radio(
    "🔗 Frequency Alignment",
    options=ALIGNMENT_MODES,
    index=0,
    key="alignment_mode",
    help=(
        "Monthly Mean aggregates Gold trading days and satellite passes onto a uniform "
        "calendar month-end grid. Satellite Pass Date pairs each instantaneous ~27-day "
        "overpass directly with the nearest Gold trading day."
    ),
)

max_uncertainty = st.sidebar.slider(
    "🎯 Max Altimetry Uncertainty (wse_u, m)",
    min_value=0.1,
    max_value=2.5,
    value=2.5,
    step=0.1,
    key="max_uncertainty",
    help=(
        "Filter out satellite passes whose estimated height uncertainty (wse_u) exceeds "
        "this threshold. Set to 2.5 m to include all available readings."
    ),
)

aligned_df, align_meta = align_datasets(
    gold_df=gold_res.df,
    euphrates_df=euphrates_res.df,
    alignment_mode=alignment_mode,
    max_uncertainty_m=max_uncertainty,
)

valid_dates = aligned_df.loc[aligned_df["Valid_Pair"], "Date"] if not aligned_df.empty else pd.Series(dtype="datetime64[ns]")
if valid_dates.empty:
    st.warning(
        "No paired observations remain under the current uncertainty threshold. "
        "Increase the Max Altimetry Uncertainty slider in the sidebar."
    )
    st.stop()

min_date = valid_dates.min().date()
max_date = valid_dates.max().date()

selected_range = st.sidebar.date_input(
    "📅 Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    key="date_range",
)

if isinstance(selected_range, (tuple, list)) and len(selected_range) == 2:
    start_date, end_date = sorted((selected_range[0], selected_range[1]))
elif isinstance(selected_range, (tuple, list)) and len(selected_range) == 1:
    start_date, end_date = selected_range[0], max_date
else:
    start_date, end_date = min_date, max_date

smoothing_window = st.sidebar.slider(
    "📈 Trend Smoothing (Periods)",
    min_value=1,
    max_value=12,
    value=3,
    key="smoothing_window",
    help="Rolling window across uniform calendar months (Monthly mode) or consecutive passes (Pass Date mode).",
)

show_normalized = st.sidebar.checkbox(
    "📊 Show Normalized Growth (Base 100%)",
    value=False,
    key="show_normalized",
    help="Re-index both series to 100% at the start of the selected date range.",
)

show_uncertainty = st.sidebar.checkbox(
    "🌫️ Show Water Uncertainty Band (±wse_u)",
    value=False,
    key="show_uncertainty",
    disabled=show_normalized,
    help="Overlay satellite altimetry uncertainty bounds on the water level curve (absolute mode only).",
)

st.sidebar.divider()
st.sidebar.subheader("📡 Data Feed Status")
eu_badge = "🟢 Live" if euphrates_res.source_mode.startswith("live") else "🟡 Fallback"
gold_badge = "🟢 Live" if gold_res.source_mode.startswith("live") else "🟡 Fallback"
st.sidebar.markdown(f"**Euphrates ({DAHITI_STATION_ID}):** {eu_badge}")
st.sidebar.caption(
    f"{euphrates_res.source_detail} · Latest: `{euphrates_res.latest_date.strftime('%Y-%m-%d')}`"
)
st.sidebar.markdown(f"**Gold (`GC=F`):** {gold_badge}")
st.sidebar.caption(
    f"{gold_res.source_detail} · Latest: `{gold_res.latest_date.strftime('%Y-%m-%d')}`"
)

if st.sidebar.button("🔄 Refresh Live Data", key="refresh_data_btn"):
    clear_live_caches()
    st.rerun()

# ─── Filtering, Smoothing & Statistics ──────────────────────────────────────
filtered = apply_window_and_smoothing(
    aligned_df=aligned_df,
    start_date=start_date,
    end_date=end_date,
    smoothing_window=smoothing_window,
    alignment_mode=alignment_mode,
)

if filtered.empty:
    st.warning(
        "No valid paired observations match the selected date range and filters. "
        "Please expand the Date Range or relax the uncertainty filter."
    )
    st.stop()

stats_summary = compute_statistics(filtered, alignment_mode=alignment_mode)

# ─── Top Metrics Row ────────────────────────────────────────────────────────
gold_start = float(filtered["Gold_Price"].iloc[0])
gold_end = float(filtered["Gold_Price"].iloc[-1])
gold_pct = ((gold_end - gold_start) / gold_start) * 100.0 if gold_start else 0.0

water_start = float(filtered["Euphrates_Level_m"].iloc[0])
water_end = float(filtered["Euphrates_Level_m"].iloc[-1])
water_pct = ((water_end - water_start) / water_start) * 100.0 if water_start else 0.0

unit_label = "Months" if alignment_mode == ALIGNMENT_MONTHLY else "Passes"
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "📅 Paired Observations",
        f"{len(filtered)} {unit_label}",
        f"{start_date.year} to {end_date.year}",
    )
with col2:
    st.metric(
        "🥇 Gold Price Change",
        f"{gold_pct:+.1f}%",
        f"${gold_end:,.0f} Window End",
    )
with col3:
    st.metric(
        "🌊 Water Level Change",
        f"{water_pct:+.1f}%",
        f"{water_end:.2f}m Window End",
        delta_color="inverse",
    )
with col4:
    if stats_summary.is_valid:
        st.metric(
            "📊 Pearson Correlation (r)",
            f"{stats_summary.pearson_r:+.2f}",
            stats_summary.correlation_label,
        )
    else:
        st.metric("📊 Pearson Correlation (r)", "N/A", "Need ≥ 3 observations")
    st.caption(
        "Measures linear level co-movement (-1 to +1); see Tab 2 for first-difference checks."
    )

st.divider()

# ─── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["📈 Dual Time Series", "🔍 Scatter & Trend", "📋 Data & Summary"]
)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: Dual Time Series
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    fig1 = build_time_series_chart(
        df=filtered,
        show_normalized=show_normalized,
        show_uncertainty=show_uncertainty,
        start_date=start_date,
        end_date=end_date,
        smoothing_window=smoothing_window,
    )
    st.plotly_chart(fig1, width="stretch", theme=None)

    with st.expander("ℹ️ Alignment & Data Coverage Diagnostics", expanded=False):
        if alignment_mode == ALIGNMENT_MONTHLY:
            missing_str = ", ".join(align_meta.missing_months) if align_meta.missing_months else "None"
            st.markdown(
                f"""
* **Alignment Strategy:** Monthly calendar month-end (`ME`) grid spanning `{align_meta.calendar_grid_months}` months (`{align_meta.aligned_valid_rows}` paired months).
* **Missing Satellite Months (Preserved as Grid Gaps):** `{align_meta.missing_months_count}` ({missing_str}). Rolling smoothing operates on the uniform calendar grid so missing months are never collapsed across gaps.
* **Months with Multiple Passes Averaged:** `{align_meta.multi_pass_months_count}` (due to ~27-day {DAHITI_SATELLITE_MISSION} repeat orbit).
* **Passes Filtered Out by Uncertainty (`wse_u > {max_uncertainty:.1f} m`):** `{align_meta.dropped_uncertainty_rows}` of `{align_meta.raw_euphrates_rows}` raw readings.
"""
            )
        else:
            st.markdown(
                f"""
* **Alignment Strategy:** Each satellite overpass timestamp is paired directly with the nearest Gold futures trading day (`merge_asof`, tolerance ±5 days).
* **Max Calendar Match Lag:** `{align_meta.max_match_gap_days}` day(s) (weekend/holiday offset).
* **Passes Retained:** `{align_meta.aligned_valid_rows}` of `{align_meta.raw_euphrates_rows}` raw readings (`{align_meta.dropped_uncertainty_rows}` filtered by `wse_u > {max_uncertainty:.1f} m`).
"""
            )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: Scatter, Regression & Honest Statistical Diagnostics
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    if not stats_summary.is_valid:
        st.info(stats_summary.invalid_reason)
    else:
        col_plot, col_info = st.columns([3, 1])

        with col_plot:
            fig2 = build_scatter_regression_chart(filtered, stats_summary)
            st.plotly_chart(fig2, width="stretch", theme=None)

        with col_info:
            st.subheader("Statistical Fit")
            diff_str = (
                f"`{stats_summary.diff_pearson_r:+.3f}` (`p={stats_summary.diff_pearson_p:.3f}`, n={stats_summary.diff_n})"
                if stats_summary.diff_n >= 3 and not pd.isna(stats_summary.diff_pearson_r)
                else "`N/A` (insufficient consecutive pairs)"
            )
            st.markdown(
                f"""
* **Pearson (r):** `{stats_summary.pearson_r:+.3f}` (`p={stats_summary.pearson_p:.2e}`)
* **Spearman (ρ):** `{stats_summary.spearman_rho:+.3f}` (`p={stats_summary.spearman_p:.2e}`)
* **R² (Level Fit):** `{stats_summary.r_squared:.3f}`
* **OLS Slope:** `{stats_summary.ols_slope:+,.2f} $/m`
* **First-Diff Δr:** {diff_str}
"""
            )
            if stats_summary.pearson_r <= -0.5:
                st.success("✅ Strong Inverse Level Association")
            elif stats_summary.pearson_r >= 0.5:
                st.warning("📈 Strong Positive Level Association")
            elif stats_summary.pearson_r <= -0.2:
                st.info("📉 Moderate Inverse Level Association")
            elif stats_summary.pearson_r >= 0.2:
                st.info("📈 Moderate Positive Level Association")
            else:
                st.info("⚖️ Weak / Negligible Linear Association")

        # Dynamic Data Storytelling & Statistical Caveat Card
        slope_direction = (
            "appreciation" if stats_summary.ols_slope < 0 else "decline"
        )
        diff_note = (
            f"However, once secular trends are removed via **consecutive first differences**, "
            f"the period-to-period correlation is **`{stats_summary.diff_pearson_r:+.2f}`** "
            f"(`p = {stats_summary.diff_pearson_p:.3f}` across `{stats_summary.diff_n}` consecutive pairs)"
            if stats_summary.diff_n >= 3 and not pd.isna(stats_summary.diff_pearson_r)
            else "Too few consecutive pairs remain in this window to compute a reliable first-difference check"
        )
        st.info(
            f"""
💡 **Analytical Interpretation & Methodological Caveat:**

* **Level Co-Movement ({start_date.year} – {end_date.year}):** Across the selected window (`n = {stats_summary.n_obs}`), the paired levels exhibit a **{stats_summary.correlation_label.lower()}** linear relationship (*Pearson `r = {stats_summary.pearson_r:+.2f}`*, *Spearman `ρ = {stats_summary.spearman_rho:+.2f}`*).

* **OLS Sensitivity:** Each 1-meter drop in Euphrates water elevation corresponds in level-space to an average Gold price {slope_direction} of **`${abs(stats_summary.ols_slope):,.2f}/oz`** (*`R² = {stats_summary.r_squared:.1%}`* of level variance).

* **Non-Stationarity Check (Spurious Trend Guard):** Gold prices exhibit a strong multi-year upward trend while river hydrology follows seasonal and multi-year drought cycles. {diff_note} — demonstrating that level correlation reflects long-term macro vs. hydrological divergence rather than a direct short-term causal link.
"""
        )

        rolling_df = compute_rolling_correlation(filtered, window=ROLLING_CORR_WINDOW)
        if not rolling_df.empty:
            fig_roll = build_rolling_correlation_chart(rolling_df, window=ROLLING_CORR_WINDOW)
            st.plotly_chart(fig_roll, width="stretch", theme=None)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: Data Summary & Export
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📊 Annual Aggregated Summary")
    yearly_summary = build_annual_summary(filtered)
    st.dataframe(yearly_summary, width="stretch")

    with st.expander("🔎 Aligned Observations Table", expanded=False):
        display_df = filtered[
            [
                "Date",
                "Gold_Price",
                "Euphrates_Level_m",
                "WSE_Uncertainty_m",
                "Pass_Count",
                "Gold_Trading_Days",
                "Alignment_Lag_Days",
            ]
        ].copy()
        display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
        display_df = display_df.rename(
            columns={
                "Gold_Price": "Gold (USD/oz)",
                "Euphrates_Level_m": "Water Level (m)",
                "WSE_Uncertainty_m": "Uncertainty ±(m)",
                "Pass_Count": "Passes",
                "Gold_Trading_Days": "Gold Days",
                "Alignment_Lag_Days": "Match Lag (d)",
            }
        )
        st.dataframe(display_df.round(3), width="stretch", hide_index=True)

    export_df = filtered[
        [
            "Date",
            "Gold_Price",
            "Euphrates_Level_m",
            "WSE_Uncertainty_m",
            "Pass_Count",
            "Gold_Trading_Days",
            "Alignment_Lag_Days",
        ]
    ].copy()
    export_df["Alignment_Mode"] = alignment_mode
    csv_data = export_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Merged Dataset (CSV)",
        data=csv_data,
        file_name=f"gold_vs_euphrates_{start_date}_{end_date}.csv",
        mime="text/csv",
        key="download_csv_btn",
    )

# ─── Footer ─────────────────────────────────────────────────────────────────
st.caption(
    f"Data Sources: DAHITI Satellite Altimetry (Station {DAHITI_STATION_ID}, DGFI-TUM · {DAHITI_SATELLITE_MISSION}) "
    "| Gold Futures via Yahoo Finance (`GC=F`) with bundled local fallback snapshots."
)
