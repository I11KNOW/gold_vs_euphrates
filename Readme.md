# 🌊 Gold Price vs. Euphrates River Water Level (2016–2026)

An interactive Exploratory Data Analysis (EDA) and monitoring dashboard investigating macro-economic valuation trends (Gold Futures) alongside real-world satellite radar altimetry measurements of the Euphrates River.

---

## 📌 Project Overview
This project explores the decade-long empirical relationship between global Gold prices ($/oz) and water surface height fluctuations of the Euphrates River (Station 41518, Iraq) between May 2016 and 2026.

> **Contextual Hook:** Inspired by the prophetic narration highlighting the historical and economic significance of the Euphrates basin, this dashboard applies modern time-series and regression techniques to public environmental and financial datasets.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.10+
* **Data Wrangling:** `pandas`, `numpy`
* **Time-Series Extraction:** `yfinance` (Yahoo Finance API)
* **Statistical Modeling:** `scipy.stats` (Pearson Correlation, OLS Linear Regression)
* **Interactive Visualization:** `plotly.graph_objects`
* **Web UI Framework:** `streamlit`

---

## 📊 Key Features
* **Dual-Axis Time Series:** Interactive synchronization comparing Gold valuation surges against river water levels.
* **Trend Smoothing Engine:** Adjustable rolling moving average (1 to 12 months) for seasonal noise filtering.
* **Regression & Scatter Analysis:** Dynamic computation of $R^2$, Slope, and $P$-value.
* **Data Export:** Integrated capability to filter by date range and download the merged dataset as CSV.

---

## 🛰️ Data Sources
1. **Euphrates River Water Surface Height (WSE):**
   * Source: **DAHITI** (Database for Hydrological Time Series of Inland Waters) – *Deutsches Geodätisches Forschungsinstitut der Technischen Universität München (DGFI-TUM)*.
   * Station ID: 41518 (Sentinel-3 radar altimetry).
2. **Gold Spot / Futures Price:**
   * Source: **Yahoo Finance** (`GC=F` continuous contracts).

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/gold-vs-euphrates-dashboard.git
   cd gold-vs-euphrates-dashboard