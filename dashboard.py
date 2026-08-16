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

# ─── إعدادات الصفحة ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold Price vs Euphrates Water Level (2016-2026)",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── العنوان والمقدمة ────────────────────────────────────────────────────────
st.title("🥇 أسعار الذهب vs 🌊 منسوب مياه نهر الفرات")
st.markdown(
    """
داشبورد تفاعلية بتقارن بين سعر أونصة الذهب وبيانات الأقمار الصناعية لمنسوب مياه نهر الفرات (محطة 41518 - العراق) من 2016 لحد 2026.

---

**قال رسول الله -صلى الله عليه وسلم-: (يوشك الفرات أن يحسر عن كنز من ذهب، فمن حضره فلا يأخذ منه شيئا) متفق عليه**
"""
)
st.divider()

# ─── تحميل ودمج البيانات ─────────────────────────────────────────────────────
@st.cache_data
def load_and_merge_data():
    csv_file = r"C:\Users\IT\Desktop\euphrates_data.csv"
    
    if not os.path.exists(csv_file):
        if os.path.exists("euphrates_data.csv"):
            csv_file = "euphrates_data.csv"
        else:
            st.error(f"❌ الملف مش موجود في المسار: {csv_file}")
            st.stop()

    # 1. قراءة بيانات الفرات
    df_euphrates = pd.read_csv(csv_file, sep=";")
    df_euphrates['Date'] = pd.to_datetime(df_euphrates['datetime'])
    
    try:
        euphrates_monthly = df_euphrates.set_index('Date')[['wse']].resample('ME').mean()
    except Exception:
        euphrates_monthly = df_euphrates.set_index('Date')[['wse']].resample('M').mean()
        
    euphrates_monthly.columns = ['Euphrates_Level_m']

    # 2. سحب أسعار الذهب من ياهو فاينانس
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

# ─── الفلاتر والتحكم الجانبي ────────────────────────────────────────────────
st.sidebar.header("⚙️ الفلاتر والتحكم")

min_date = df['Date'].min().date()
max_date = df['Date'].max().date()

selected_range = st.sidebar.date_input(
    "📅 الفترة الزمنية",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(selected_range, tuple) and len(selected_range) == 2:
    start_date, end_date = selected_range
else:
    start_date, end_date = min_date, max_date

# تنعيم البيانات
smoothing_window = st.sidebar.slider("📈 تنعيم المنحنى (بالمتوسط المتحرك - شهور)", min_value=1, max_value=12, value=3)

# الفلترة
mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
filtered = df.loc[mask].copy()

# تطبيق التنعيم
if smoothing_window > 1:
    filtered['Gold_Plot'] = filtered['Gold_Price'].rolling(smoothing_window, min_periods=1).mean()
    filtered['Euphrates_Plot'] = filtered['Euphrates_Level_m'].rolling(smoothing_window, min_periods=1).mean()
else:
    filtered['Gold_Plot'] = filtered['Gold_Price']
    filtered['Euphrates_Plot'] = filtered['Euphrates_Level_m']

# ─── كروت الأرقام والمؤشرات الرئيسية ─────────────────────────────────────────
gold_start = filtered['Gold_Price'].iloc[0]
gold_end = filtered['Gold_Price'].iloc[-1]
gold_pct = ((gold_end - gold_start) / gold_start) * 100

water_start = filtered['Euphrates_Level_m'].iloc[0]
water_end = filtered['Euphrates_Level_m'].iloc[-1]
water_pct = ((water_end - water_start) / water_start) * 100

pearson_val, _ = stats.pearsonr(filtered['Gold_Price'], filtered['Euphrates_Level_m'])

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📅 حجم البيانات", f"{len(filtered)} شهر", f"من {start_date.year} لـ {end_date.year}")
with col2:
    st.metric("🥇 تغير سعر الذهب", f"{gold_pct:+.1f}%", f"${gold_end:,.0f} السعر الحالي")
with col3:
    st.metric("🌊 تغير منسوب المياه", f"{water_pct:+.1f}%", f"{water_end:.2f}م المنسوب الحالي", delta_color="inverse")
with col4:
    rel_text = "علاقة عكسية قوية جداً" if pearson_val < -0.7 else ("علاقة عكسية واضحة" if pearson_val < -0.5 else "علاقة عكسية متوسطة")
    st.metric("📊 معامل الارتباط (بيرسون)", f"{pearson_val:.2f}", rel_text)
    st.caption("ℹ️ بيقيس قوة واتجاه العلاقة: كل ما يقرب من -1 معناه إن لما المياه بتقل، الذهب بيزيد.")

st.divider()

# ─── التبويبات الرئيسية ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 الرسم الزمني المزدوج", "🔍 تحليل الانحدار والارتباط", "📋 ملخص البيانات والتحميل"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: الرسم الزمني المزدوج
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    fig1 = go.Figure()
    
    # مسار الذهب
    fig1.add_trace(go.Scatter(
        x=filtered['Date'],
        y=filtered['Gold_Plot'],
        name="سعر الذهب (بالدولار للأونصة)",
        line=dict(color="#FFB300", width=3),
        yaxis="y1",
        hovertemplate="<b>التاريخ:</b> %{x|%b %Y}<br><b>الذهب:</b> $%{y:,.1f}<extra></extra>"
    ))
    
    # مسار منسوب مياه الفرات
    fig1.add_trace(go.Scatter(
        x=filtered['Date'],
        y=filtered['Euphrates_Plot'],
        name="منسوب مياه الفرات (متر فوق سطح البحر)",
        line=dict(color="#0077B6", width=2.5),
        yaxis="y2",
        hovertemplate="<b>التاريخ:</b> %{x|%b %Y}<br><b>المنسوب:</b> %{y:.2f} متر<extra></extra>"
    ))
    
    fig1.update_layout(
        title=dict(text=f"مقارنة مسار أسعار الذهب مع منسوب نهر الفرات ({start_date.year} – {end_date.year})"),
        template="plotly_white",
        height=520,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(
            title=dict(text="سعر الذهب ($)", font=dict(color="#FFB300")),
            tickfont=dict(color="#FFB300"),
            side="left"
        ),
        yaxis2=dict(
            title=dict(text="ارتفاع منسوب المياه (متر)", font=dict(color="#0077B6")),
            tickfont=dict(color="#0077B6"),
            overlaying="y",
            side="right",
            showgrid=False
        ),
        xaxis=dict(title=dict(text="التاريخ"))
    )
    st.plotly_chart(fig1, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: تحليل الانحدار والارتباط
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    col_plot, col_info = st.columns([3, 2])
    
    x_data = filtered['Euphrates_Level_m']
    y_data = filtered['Gold_Price']
    slope, intercept, r_val, p_val, std_err = stats.linregress(x_data, y_data)
    
    x_fit = np.linspace(x_data.min(), x_data.max(), 100)
    y_fit = slope * x_fit + intercept
    r2 = r_val ** 2
    
    with col_plot:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers',
            name='قراءات الشهور',
            marker=dict(size=8, color='#0077B6', opacity=0.7, line=dict(width=1, color='white')),
            hovertemplate="<b>المنسوب:</b> %{x:.2f} م<br><b>الذهب:</b> $%{y:,.1f}<extra></extra>"
        ))
        fig2.add_trace(go.Scatter(
            x=x_fit,
            y=y_fit,
            mode='lines',
            name=f'خط الاتجاه العام (الميل: {slope:.1f})',
            line=dict(color='#E63946', width=2.5)
        ))
        fig2.update_layout(
            title=dict(text="العلاقة الخطية بين منسوب المياه وسعر الذهب"),
            xaxis=dict(title=dict(text="منسوب مياه الفرات (متر)")),
            yaxis=dict(title=dict(text="سعر أونصة الذهب ($)")),
            template="plotly_white",
            height=460
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_info:
        st.subheader("📊 المؤشرات الإحصائية")
        st.markdown(f"""
        * **معامل التحديد ($R^2$):** `{r2:.3f}` *(المنسوب بيفسر حوالي {r2*100:.1f}% من حركة الذهب)*
        * **ميل الخط (Slope):** `{slope:.2f}`
        * **الدلالة الإحصائية (P-value):** `{p_val:.4e}`
        """)

        # شرح بسيط وعامي للنتيجة
        st.success(
            f"""
            💡 **قراءة الداتا ببساطة (Data Story):**
            * **العلاقة العكسية واضحة جداً:** معامل الارتباط طالع **`{pearson_val:.2f}`**، وده معناه إنه طول الفترة من ({start_date.year} لـ {end_date.year})، كل ما مياه الفرات كانت بتقل وبتنحسر، أسعار الذهب عالمياً كانت بتطير لفوق وبتسجل أرقام تاريخية.
            * **حسبة الانحدار الخطي:** إحصائياً، كل **انخفاض بمقدار متر واحد** في منسوب مياه الفرات كان بيقابله في المتوسط **زيادة بحوالي ${abs(slope):,.2f}** في سعر أونصة الذهب.
            """
        )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: ملخص البيانات والتحميل
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📊 ملخص الأداء السنوي")
    yearly_summary = filtered.groupby('Year').agg(
        متوسط_الذهب=('Gold_Price', 'mean'),
        أقل_سعر_ذهب=('Gold_Price', 'min'),
        أعلى_سعر_ذهب=('Gold_Price', 'max'),
        متوسط_المنسوب=('Euphrates_Level_m', 'mean'),
        أقل_منسوب=('Euphrates_Level_m', 'min'),
        أعلى_منسوب=('Euphrates_Level_m', 'max')
    ).round(2)
    st.dataframe(yearly_summary, use_container_width=True)

    csv_data = filtered[['Date', 'Gold_Price', 'Euphrates_Level_m']].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 تحميل البيانات المدمجة (CSV)",
        data=csv_data,
        file_name=f"gold_vs_euphrates_{start_date}_{end_date}.csv",
        mime="text/csv"
    )

# ─── الفوتر ومصادر البيانات ──────────────────────────────────────────────────
st.caption("مصادر الداتا: قياسات الأقمار الصناعية لنهر الفرات DAHITI (محطة 41518، DGFI-TUM) | أسعار عقود الذهب عبر Yahoo Finance (`GC=F`).")
