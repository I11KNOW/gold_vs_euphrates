# 🌊 Gold Price vs. Euphrates River Water Level (2016–2026)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive Exploratory Data Analysis (EDA) dashboard investigating the correlation between **Gold Price** and **Euphrates Water Level** from 2016 to 2026.

---

## 📌 Project Overview
This dashboard performs real-time data merging, statistical analysis, and visualization to compare macro-economic gold trends with hydrological changes in the Euphrates River basin.

> **Contextual Hook:** Inspired by the prophetic narration highlighting the significance of the Euphrates basin, this dashboard applies modern time-series and regression techniques to environmental and financial datasets.
>
> **قال رسول الله -صلى الله عليه وسلم-:** *(يوشك الفرات أن يحسر عن كنز من ذهب، فمن حضره فلا يأخذ منه شيئا)* — **متفق عليه**

---

## 🛠️ Tech Stack & Libraries
* **Framework:** Streamlit
* **Data Wrangling:** `pandas`, `numpy`
* **Financial Data:** `yfinance`
* **Statistical Modeling:** `scipy.stats` (Pearson Correlation, OLS Linear Regression)
* **Interactive Visualization:** `plotly.graph_objects`

---

## 📊 Key Features
* **Dual-Axis Time Series:** Visualizes Gold prices and water heights with configurable rolling trend smoothing (1–12 months).
* **Statistical Insights:** Automatic computation of **Pearson correlation ($r$)**, **$R^2$ determination**, **Slope**, and **$P$-value**.
* **Normalized Metrics:** Includes normalized growth tracking (Base 100%) for accurate comparison.
* **Data Export:** Built-in capability to filter by date and download the merged dataset as a CSV.
* **Aggregated Summary:** Generates annual statistical summaries (Mean, Min, Max) for both datasets.

---

## 🛰️ Data Sources
1. **Euphrates River (WSE):** DAHITI (Station 41518, DGFI-TUM).
2. **Gold Price:** Yahoo Finance (`GC=F` futures contract).

---

## 🚀 How to Run Locally

### 1. Clone the repository:
```bash
git clone https://github.com/<your-username>/gold-vs-euphrates-dashboard.git
cd gold-vs-euphrates-dashboard
```

### 2. Install Dependencies:
```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset:
Ensure your `euphrates_data.csv` file (semicolon-delimited with `datetime` and `wse` columns) is in the root directory.

### 4. Launch the Dashboard:
```bash
streamlit run dashboard.py
```

---

## 📁 Repository Structure
```text
├── dashboard.py           # Main application logic
├── euphrates_data.csv     # Satellite water level dataset
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

---

## 📄 License
Distributed under the MIT License.
