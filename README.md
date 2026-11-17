# Gold Price vs. Euphrates River Water Level (2016–2026)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%20--%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive Exploratory Data Analysis (EDA) dashboard investigating the empirical relationship between **COMEX Gold Futures (`GC=F`)** and **Euphrates River Water Surface Elevation** (**DAHITI Station 41518, Iraq**) from 2016 through 2026.

---

## 📌 Project Overview

This dashboard combines live financial market feeds and satellite radar altimetry measurements with resilient local fallbacks, frequency alignment controls, and transparent time-series diagnostics.

> **Contextual Inspiration:** Inspired by the prophetic narration highlighting the Euphrates River and gold, this project applies rigorous time-series alignment and statistical analysis to environmental and macro-financial datasets.
>
> **قال رسول الله -صلى الله عليه وسلم-:** *(يوشك الفرات أن يحسر عن كنز من ذهب، فمن حضره فلا يأخذ منه شيئا)* — **متفق عليه**

---

## 🛰️ Data Sources & Resilience Pipeline

| Dataset | Upstream Source | Frequency & Coverage | Caching TTL | Local Fallback |
| :--- | :--- | :--- | :--- | :--- |
| **Euphrates Water Level (`wse`, `wse_u`)** | [DAHITI Station 41518](https://dahiti.dgfi.tum.de/en/41518/) (DGFI-TUM · Iraq, `31.5545°N, 44.6587°E`, Sentinel-3A Pass `0212*`) | ~27-day orbital passes (`2016-05-13` – `2026-10-15`, 136 readings) | `24h` | [`data/euphrates_data.csv`](data/euphrates_data.csv) |
| **Gold Futures (`GC=F`)** | Yahoo Finance via `yfinance` (with Chart v8 JSON secondary fallback) | Daily trading sessions (`2016-05-02` – `2026-11-16`) | `6h` | [`data/gold_prices_fallback.csv`](data/gold_prices_fallback.csv) |

### Resilience Design
* **Multi-tier ingestion:** Euphrates data checks `DAHITI_API_KEY` (official DAHITI API v2) if configured, otherwise fetches DAHITI's public Station 41518 altimetry chart feed, and automatically falls back to `data/euphrates_data.csv` if offline or unreachable. Gold fetches via `yfinance` (with a hard non-blocking timeout guard), falls back to Yahoo Chart v8 JSON, and then to `data/gold_prices_fallback.csv`.
* **Failure-safe caching:** `@st.cache_data` caches only successful live responses so transient network outages or rate limits never stick in cache; a sidebar **Refresh Live Data** button clears caches on demand.

---

## 📐 Methodology & Statistical Alignment

Comparing daily financial prices against ~27-day satellite altimetry passes introduces subtle traps that this codebase explicitly handles:

1. **Frequency Alignment Modes (Sidebar Toggle):**
   * **Monthly Mean (Calendar Grid):** Clips Gold data at the latest available satellite month (preventing partial trailing months from skewing averages), resamples both series to month-end (`ME`), and **reindexes onto a complete uniform calendar grid** before computing rolling averages. Months without satellite passes (`2017-06`, `2018-09`) remain missing rather than silently distorting rolling windows.
   * **Satellite Pass Date (Nearest Trading Day):** Uses `pd.merge_asof(..., direction="nearest", tolerance=5d)` to pair each instantaneous satellite overpass directly with its nearest Gold trading day (eliminating intra-month averaging artifacts).
2. **Altimetry Uncertainty Filtering (`wse_u`):** DAHITI reports per-pass measurement uncertainty (`0.001 m` to `2.18 m`). Users can filter noisy passes via the **Max Altimetry Uncertainty** slider and toggle `±wse_u` error bands on the time-series chart.
3. **Non-Stationarity & Spurious Correlation Guard:** Gold exhibits a strong secular upward trend over 2016–2026 while river elevation fluctuates across seasonal and multi-year drought cycles. Alongside level-space metrics (**Pearson $r$**, **Spearman $\rho$**, and **OLS linear regression**), the dashboard computes **consecutive first-difference correlation ($\Delta r$)** and a **12-period rolling Pearson correlation** chart to expose regime shifts and distinguish long-term structural divergence from short-term causality.
4. **Normalized Growth (Base 100%):** Re-indexes both series dynamically to `100%` at the start of the selected date window for relative scale comparison.

---

## 📁 Project Structure

```text
gold_vs_euphrates/
├── dashboard.py                     # Streamlit UI entrypoint
├── requirements.txt                 # Exact pinned production dependencies
├── README.md                        # Methodology, sources, and deployment guide
├── .streamlit/
│   ├── config.toml                  # Theme and server configuration
│   └── secrets.toml.example         # Optional DAHITI_API_KEY template
├── data/
│   ├── euphrates_data.csv           # Bundled DAHITI Station 41518 snapshot (2016–2026)
│   └── gold_prices_fallback.csv     # Bundled Gold futures snapshot (2016–2026)
├── src/
│   ├── __init__.py
│   ├── config.py                    # Constants, URLs, paths, and cache TTLs
│   ├── data_loader.py               # Cached live fetchers and fallback loaders
│   ├── analysis.py                  # Alignment, gap-aware smoothing, and statistics
│   └── charts.py                    # Responsive Plotly visualizations
└── tests/
    ├── test_data_loader.py          # Ingestion, HTML parsing, and fallback tests
    ├── test_analysis.py             # Alignment, edge-case, and statistics tests
    └── test_app.py                  # End-to-end Streamlit AppTest UI verification
```

---

## 🚀 Local Setup & Running

### 1. Clone the repository
```bash
git clone https://github.com/I11KNOW/gold_vs_euphrates.git
cd gold_vs_euphrates
```

### 2. Create a virtual environment & install pinned dependencies
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 3. (Optional) Configure DAHITI API v2 Key
The dashboard runs out-of-the-box without credentials. If you have a registered DAHITI account and want to query API v2 directly:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml and add your DAHITI_API_KEY
```

### 4. Launch the Dashboard
```bash
streamlit run dashboard.py
```

### 5. Run the Test Suite
```bash
pip install pytest
pytest -v
```

---

## ☁️ Deploying on Streamlit Community Cloud

1. Push the updated repository to GitHub (`main` branch).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click **New app**.
3. Select repository **`I11KNOW/gold_vs_euphrates`**, branch **`main`**, and main file path **`dashboard.py`**.
4. Open **Advanced settings** and select Python **3.12** or **3.13** (to match the pinned wheels in `requirements.txt`).
5. *(Optional)* Under **Secrets**, add `DAHITI_API_KEY = "your_key"` if you want authenticated API v2 access; otherwise leave blank to use the public feed + local fallback.
6. Click **Deploy**.

---

## 📄 License
Distributed under the MIT License.
