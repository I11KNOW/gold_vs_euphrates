"""Responsive, clearly labeled Plotly visualizations for the dashboard."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.analysis import StatsSummary


def build_time_series_chart(
    df: pd.DataFrame,
    show_normalized: bool,
    show_uncertainty: bool,
    start_date: date,
    end_date: date,
    smoothing_window: int,
) -> go.Figure:
    """Create the dual-axis Gold Price vs. Euphrates Water Level time-series chart."""
    fig = go.Figure()

    y1_col = "Gold_Norm_Plot" if show_normalized else "Gold_Plot"
    y2_col = "Euphrates_Norm_Plot" if show_normalized else "Euphrates_Plot"

    y1_name = "Gold Index (Base 100%)" if show_normalized else "Gold Price (USD / oz)"
    y2_name = (
        "Euphrates Index (Base 100%)"
        if show_normalized
        else "Euphrates Water Level (m)"
    )

    y1_hover = "%{y:,.1f}%" if show_normalized else "$%{y:,.2f}"
    y2_hover = "%{y:,.2f}%" if show_normalized else "%{y:.3f} m"

    if show_uncertainty and not show_normalized and df["WSE_Uncertainty_m"].notna().any():
        unc = df["WSE_Uncertainty_m"].fillna(0.0)
        upper = df["Euphrates_Plot"] + unc
        lower = df["Euphrates_Plot"] - unc

        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                yaxis="y2",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=lower,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(0, 119, 182, 0.14)",
                name="Water Uncertainty (±wse_u)",
                hoverinfo="skip",
                yaxis="y2",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df[y1_col],
            name=y1_name,
            line=dict(color="#D97706", width=3),
            yaxis="y1",
            hovertemplate=f"<b>Gold:</b> {y1_hover}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df[y2_col],
            name=y2_name,
            line=dict(color="#0077B6", width=2.5),
            yaxis="y2",
            hovertemplate=f"<b>Euphrates:</b> {y2_hover}<extra></extra>",
        )
    )

    mode_label = "Normalized (Base 100%)" if show_normalized else "Absolute Values"
    smooth_label = f"{smoothing_window}-Period Rolling Mean" if smoothing_window > 1 else "Un-smoothed"

    fig.update_layout(
        title=dict(
            text=(
                f"Gold vs. Euphrates ({start_date.year} – {end_date.year})"
                f"<br><sup>{mode_label} | {smooth_label}</sup>"
            ),
            x=0.0,
            xanchor="left",
        ),
        template="plotly_white",
        height=540,
        margin=dict(l=60, r=60, t=80, b=95),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="center",
            x=0.5,
        ),
        yaxis=dict(
            title=dict(text=y1_name, font=dict(color="#B45309")),
            tickfont=dict(color="#B45309"),
            side="left",
        ),
        yaxis2=dict(
            title=dict(text=y2_name, font=dict(color="#0077B6")),
            tickfont=dict(color="#0077B6"),
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        xaxis=dict(title=dict(text="Date")),
    )
    return fig


def build_scatter_regression_chart(
    df: pd.DataFrame,
    stats_summary: StatsSummary,
) -> go.Figure:
    """Create the Water Level vs. Gold Price scatter plot with chronological color and OLS fit."""
    fig = go.Figure()

    date_labels = df["Date"].dt.strftime("%Y-%m-%d")
    unc_labels = df["WSE_Uncertainty_m"].fillna(0.0)
    customdata = np.stack([date_labels, unc_labels], axis=-1)

    fig.add_trace(
        go.Scatter(
            x=df["Euphrates_Level_m"],
            y=df["Gold_Price"],
            mode="markers",
            name="Aligned Observations",
            customdata=customdata,
            marker=dict(
                size=9,
                color=df["Year"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Year", thickness=14),
                opacity=0.85,
                line=dict(width=1, color="white"),
            ),
            hovertemplate=(
                "<b>Date:</b> %{customdata[0]}<br>"
                "<b>Water Level:</b> %{x:.3f} m (±%{customdata[1]:.3f} m)<br>"
                "<b>Gold Price:</b> $%{y:,.2f}/oz<extra></extra>"
            ),
        )
    )

    if stats_summary.is_valid:
        x_vals = df["Euphrates_Level_m"]
        x_fit = np.linspace(x_vals.min(), x_vals.max(), 100)
        y_fit = stats_summary.ols_slope * x_fit + stats_summary.ols_intercept
        fig.add_trace(
            go.Scatter(
                x=x_fit,
                y=y_fit,
                mode="lines",
                name=f"OLS Trend Fit (Slope: {stats_summary.ols_slope:+,.1f} $/m)",
                line=dict(color="#E63946", width=2.5),
            )
        )

    fig.update_layout(
        title=dict(
            text="Water Level vs. Gold Price (OLS Fit)",
            x=0.0,
            xanchor="left",
        ),
        xaxis=dict(title=dict(text="Euphrates Water Surface Elevation (m)")),
        yaxis=dict(title=dict(text="Gold Price (USD / oz)")),
        template="plotly_white",
        height=490,
        margin=dict(l=60, r=40, t=65, b=95),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
        ),
    )
    return fig


def build_rolling_correlation_chart(
    rolling_df: pd.DataFrame,
    window: int,
) -> go.Figure:
    """Plot rolling Pearson correlation to highlight regime changes over time."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=rolling_df["Date"],
            y=rolling_df["Rolling_Pearson_r"],
            mode="lines",
            name=f"{window}-Period Rolling Pearson r",
            line=dict(color="#0F766E", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(15, 118, 110, 0.12)",
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Rolling r:</b> %{y:+.3f}<extra></extra>",
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#64748B",
        annotation_text="Zero Correlation (r = 0)",
        annotation_position="top left",
    )

    fig.update_layout(
        title=dict(
            text=f"Rolling {window}-Period Pearson Correlation",
            x=0.0,
            xanchor="left",
        ),
        xaxis=dict(title=dict(text="Date")),
        yaxis=dict(title=dict(text="Pearson Correlation (r)"), range=[-1.05, 1.05]),
        template="plotly_white",
        height=320,
        margin=dict(l=60, r=30, t=60, b=45),
        showlegend=False,
    )
    return fig
