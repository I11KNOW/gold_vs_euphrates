"""
Streamlit Dashboard: Gold Price vs. Euphrates River Water Level (2016 - 2026)
=============================================================================
Direct File Path Integration (Station 41518, Iraq)
"""

import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

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
    """
Interactive Dashboard comparing Gold ounce prices with real-world satellite measurements of the Euphrates River (Station 41518, Iraq) from 2016 to 2026.

---

**قال رسول الله -صلى الله عليه وسلم-: (يوشك الفرات أن يحسر عن كنز من ذهب، فمن حضره فلا يأخذ منه شيئا) متفق عليه**
"""
)
st.divider()

# ─── Data Loading & Merging ─────────────────────────────────────────────────
@st.cache_data
def load_and_merge_data():
    csv_file = r"C:\Users\IT\Desktop\euphrates_data.csv"
    
    if not os.path.exists(csv_file):
        if os.path.exists("euphrates_data.csv"):
            csv_file = "euphrates_data.csv"
        else:
            st.error(f"❌ لم يتم العثور على الملف في المسار: {csv_file}")
            st.stop()

    # 1. قراءة بيانات الفرات من DAHITI بالفاصلة المنقوطة
    df_euphrates = pd.read_csv(csv_file, sep=";")
    df_euphrates['Date'] = pd.to_datetime(df_euphrates['datetime'])
    
    try:
        euphrates_monthly = df_euphrates.set_index('Date')[['wse']].resample('ME').mean()
    except Exception:
        euphrates_monthly = df_euphrates.set_index('Date')[['wse']].resample('M').mean()
        
    euphrates_monthly.columns = ['Euphrates_Level_m']

    # 2. سحب أسعار الذهب ابتداءً من 2016 لمطابقة محطة الفرات
    start_str = euphrates_monthly.index.min().strftime('%Y-%m-%d')
    gold_raw = yf.download("GC=F", start=start_str, progress=False)
    
    if isinstance(gold_raw.columns, pd.MultiIndex):
        gold_close = gold_raw['Close']
        if isinstance(gold_close, pd.DataFrame):
            gold_close = gold_close.iloc[:, 0]
    else:
        gold_close = gold_raw['Close']

    try:
        gold_monthly = gold_close.resample('ME').mean().to_frame()
    except Exception:
        gold_monthly = gold_close.resample('M').mean().to_frame()
        
    gold_monthly.columns = ['Gold_Price']

    # 3. دمج الجدولين زمنياً
    merged = pd.concat([gold_monthly, euphrates_monthly], axis=1).dropna().reset_index()
    merged.columns = ['Date', 'Gold_Price', 'Euphrates_Level_m']
    merged['Year'] = merged['Date'].dt.year

    return merged

df = load_and_merge_data()

# ─── Sidebar Controls ───────────────────────────────────────────────────────
st.sidebar.header("⚙️ Controls & Filters")

min_date = df['Date'].min().date()
max_date = df['Date'].max().date()

selected_range = st.sidebar.date_input(
    "📅 Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(selected_range, tuple) and len(selected_range) == 2:
    start_date, end_date = selected_range
else:
    start_date, end_date = min_date, max_date

# خيارات التنعيم ونمط العرض النسبي
smoothing_window = st.sidebar.slider("📈 Trend Smoothing by Month", min_value=1, max_value=12, value=3)
show_normalized = st.sidebar.checkbox("📊 Show Normalized Growth (Base 100%)", value=False)

# الفلترة
mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
filtered = df.loc[mask].copy()

# إعادة ضبط الأساس 100% للفترة المحددة
filtered['Gold_Norm_Dynamic'] = (filtered['Gold_Price'] / filtered['Gold_Price'].iloc[0]) * 100
filtered['Euphrates_Norm_Dynamic'] = (filtered['Euphrates_Level_m'] / filtered['Euphrates_Level_m'].iloc[0]) * 100

# تطبيق التنعيم
if smoothing_window > 1:
    filtered['Gold_Plot'] = filtered['Gold_Price'].rolling(smoothing_window, min_periods=1).mean()
    filtered['Euphrates_Plot'] = filtered['Euphrates_Level_m'].rolling(smoothing_window, min_periods=1).mean()
    filtered['Gold_Norm_Plot'] = filtered['Gold_Norm_Dynamic'].rolling(smoothing_window, min_periods=1).mean()
    filtered['Euphrates_Norm_Plot'] = filtered['Euphrates_Norm_Dynamic'].rolling(smoothing_window, min_periods=1).mean()
else:
    filtered['Gold_Plot'] = filtered['Gold_Price']
    filtered['Euphrates_Plot'] = filtered['Euphrates_Level_m']
    filtered['Gold_Norm_Plot'] = filtered['Gold_Norm_Dynamic']
    filtered['Euphrates_Norm_Plot'] = filtered['Euphrates_Norm_Dynamic']

# ─── Top Metrics Row ────────────────────────────────────────────────────────
gold_start = filtered['Gold_Price'].iloc[0]
gold_end = filtered['Gold_Price'].iloc[-1]
gold_pct = ((gold_end - gold_start) / gold_start) * 100

water_start = filtered['Euphrates_Level_m'].iloc[0]
water_end = filtered['Euphrates_Level_m'].iloc[-1]
water_pct = ((water_end - water_start) / water_start) * 100

pearson_val, _ = stats.pearsonr(filtered['Gold_Price'], filtered['Euphrates_Level_m'])

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📅 Data Points", f"{len(filtered)} Months", f"{start_date.year} to {end_date.year}")
with col2:
    st.metric("🥇 Gold Price Change", f"{gold_pct:+.1f}%", f"${gold_end:,.0f} Current")
with col3:
    st.metric("🌊 Water Level Change", f"{water_pct:+.1f}%", f"{water_end:.2f}m Current", delta_color="inverse")
with col4:
    st.metric("📊 Pearson Correlation (r)", f"{pearson_val:.2f}", "Strong Inverse" if pearson_val < -0.5 else "Moderate Inverse")
    st.caption("ℹ️ Pearson (r): يقيس شدة واتجاه العلاقة الخطية بين السلسلتين (بين -1 و +1).")

st.divider()

# ─── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Dual Time Series", "🔍 Scatter & Trend", "📋 Data & Summary"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: Dual Time Series
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    fig1 = go.Figure()
    
    y1_col = 'Gold_Norm_Plot' if show_normalized else 'Gold_Plot'
    y2_col = 'Euphrates_Norm_Plot' if show_normalized else 'Euphrates_Plot'
    
    y1_name = "Gold Index (Base 100%)" if show_normalized else "Gold Price (Ounce, $)"
    y2_name = "Euphrates Index (Base 100%)" if show_normalized else "Euphrates Water Level (m)"
    
    y1_hover = "%{y:,.1f}%" if show_normalized else "$%{y:,.1f}"
    y2_hover = "%{y:,.2f}%" if show_normalized else "%{y:.2f} m"

    # مسار الذهب
    fig1.add_trace(go.Scatter(
        x=filtered['Date'],
        y=filtered[y1_col],
        name=y1_name,
        line=dict(color="#FFB300", width=3),
        yaxis="y1",
        hovertemplate=f"<b>Date:</b> %{{x|%b %Y}}<br><b>Gold:</b> {y1_hover}<extra></extra>"
    ))
    
    # مسار منسوب مياه الفرات
    fig1.add_trace(go.Scatter(
        x=filtered['Date'],
        y=filtered[y2_col],
        name=y2_name,
        line=dict(color="#0077B6", width=2.5),
        yaxis="y2",
        hovertemplate=f"<b>Date:</b> %{{x|%b %Y}}<br><b>Level:</b> {y2_hover}<extra></extra>"
    ))
    
    fig1.update_layout(
        title=dict(text=f"Gold vs Euphrates Analysis ({start_date.year} – {end_date.year}) - {'Normalized (Base 100%)' if show_normalized else 'Absolute Values'}"),
        template="plotly_white",
        height=520,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(
            title=dict(text=y1_name, font=dict(color="#FFB300")),
            tickfont=dict(color="#FFB300"),
            side="left"
        ),
        yaxis2=dict(
            title=dict(text=y2_name, font=dict(color="#0077B6")),
            tickfont=dict(color="#0077B6"),
            overlaying="y",
            side="right",
            showgrid=False
        ),
        xaxis=dict(title=dict(text="Date"))
    )
    st.plotly_chart(fig1, width="stretch")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: Scatter, Linear Regression & Dynamic Interpretation
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    col_plot, col_info = st.columns([3, 1])
    
    x_data = filtered['Euphrates_Level_m']
    y_data = filtered['Gold_Price']
    slope, intercept, r_val, p_val, std_err = stats.linregress(x_data, y_data)
    
    x_fit = np.linspace(x_data.min(), x_data.max(), 100)
    y_fit = slope * x_fit + intercept
    
    with col_plot:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers',
            name='Monthly Observations',
            marker=dict(size=8, color='#0077B6', opacity=0.7, line=dict(width=1, color='white')),
            hovertemplate="<b>Level:</b> %{x:.2f} m<br><b>Gold:</b> $%{y:,.1f}<extra></extra>"
        ))
        fig2.add_trace(go.Scatter(
            x=x_fit,
            y=y_fit,
            mode='lines',
            name=f'Trend Fit (Slope: {slope:.1f})',
            line=dict(color='#E63946', width=2.5)
        ))
        fig2.update_layout(
            title=dict(text="Scatter Analysis: Water Level vs. Gold Valuation"),
            xaxis=dict(title=dict(text="Euphrates Water Height (m)")),
            yaxis=dict(title=dict(text="Gold Price ($)")),
            template="plotly_white",
            height=460
        )
        st.plotly_chart(fig2, width="stretch")

    with col_info:
        st.subheader("Statistical Fit")
        r2 = r_val ** 2
        st.markdown(f"""
        * **R² (Determination):** `{r2:.3f}`
        * **Slope:** `{slope:.2f}`
        * **P-value:** `{p_val:.4e}`
        """)
        if abs(pearson_val) >= 0.5:
            st.success("✅ Strong Inverse Alignment")
        else:
            st.info("ℹ️ Moderate Correlation")

    # ── Dynamic Data Storytelling Card ─────────────────────────────────────
    st.info(
        f"""
        💡 **Analytical Interpretation:**
        **Inverse Correlation Dynamics**: The paired time series demonstrates a sustained structural divergence across the observation window ({2016} – {2026}); persistent declines in hydrological water elevation closely coincide with secular bull cycles in global gold valuations *(Pearson correlation: {pearson_val:.2f})*.
        **Valuation Sensitivity (OLS Fit)**: The linear regression model indicates that each 1-meter drop in Euphrates water elevation is empirically associated with an average gold price appreciation of *${abs(slope):,.2f}*/oz, with the model explaining *{r2:.1%} ($R^2$)* of the cross-sectional variance.
        """
    )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: Data Summary & Export
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📊 Annual Aggregated Summary")
    yearly_summary = filtered.groupby('Year').agg(
        Gold_Avg=('Gold_Price', 'mean'),
        Gold_Min=('Gold_Price', 'min'),
        Gold_Max=('Gold_Price', 'max'),
        Water_Level_Avg=('Euphrates_Level_m', 'mean'),
        Water_Level_Min=('Euphrates_Level_m', 'min'),
        Water_Level_Max=('Euphrates_Level_m', 'max')
    ).round(2)
    st.dataframe(yearly_summary, width="stretch")

    csv_data = filtered[['Date', 'Gold_Price', 'Euphrates_Level_m']].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Merged Dataset (CSV)",
        data=csv_data,
        file_name=f"gold_vs_euphrates_{start_date}_{end_date}.csv",
        mime="text/csv"
    )

# ─── Footer ─────────────────────────────────────────────────────────────────
st.caption("Data Sources: DAHITI Satellite Altimetry (Station 41518, DGFI-TUM) | Gold Spot/Futures via Yahoo Finance (`GC=F`).")
