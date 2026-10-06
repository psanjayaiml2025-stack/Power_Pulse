# PowerPulse Analytics
### Explainable Energy Decision Intelligence for Indian Households
*Smart-Meter Analytics, Forecasting, Behavioural Profiling, Anomaly Detection and Scenario-Based Energy Optimization*

## Features

- **Consumption DNA** — an explainable behavioural profile (e.g. "Evening-Peak Consumer") derived from documented, fixed numeric thresholds applied to real computed statistics (peak-hour concentration, night share, day-to-day variability, weekend deviation, anomaly rate). No trait is assigned without a visible reason.
- **PowerPulse Energy Health Score** — a 0-100 score built from four transparent, weighted components (baseline alignment, anomaly control, peak balance, trend outlook), always shown with its full breakdown.
- **Forecasting** — RandomForestRegressor vs. a persistence baseline, chronological train/test split (no leakage), MAE/RMSE/MAPE, and an 80%-style expected range derived from actual backtest residuals (only shown when there's enough held-out data to calculate it — never fabricated).
- **Anomaly Detection** — rolling z-score cross-checked against Isolation Forest, each anomaly reported with actual value, expected range, deviation %, and severity.
- **Behavioural Clustering** — unsupervised K-Means over per-meter/building consumption features, with silhouette-score-based selection of the number of clusters (auditable, not a fixed guess).
- **Building/Meter Priority Ranking** — for College/Campus and Institution modes, ranks meters/buildings by consumption share, growth trend, and anomaly count.
- **What-If Simulator** — scenario-based savings projection (peak/off-peak/hours reduction), always labeled as a simulation, never a guaranteed outcome. Recommendations link directly into a prefilled scenario ("Simulate this intervention").
- **Official CEA Benchmark** — national FY 2023-24 vs. provisional FY 2024-25 electricity statistics, kept strictly separate from household-level ML training data.
- **Reports** — printable/exportable summary combining dashboard, forecast, anomalies, and priority ranking.
- Meter/bill upload, manual reading entry, EDA with hour×day heatmap, tariff-based cost estimation, dark/light theme, campus/institution/household environment modes.

## Architecture

```
React (Vite) → FastAPI → Data Science Pipeline → ML → Decision Support
```
SQLite + SQLAlchemy for metadata/history (no PostgreSQL, no Docker, no external services).

## Data Science Pipeline

Data Collection → Validation → Cleaning → Transformation → EDA → Feature Engineering →
Statistical Analysis → Machine Learning → Model Evaluation → Forecasting → Anomaly Detection →
Behavioural Clustering → Explainable Insights → Scenario Simulation → Recommendations →
Decision Support → Official Benchmarking → Reporting

## Dataset

**Primary (ML/analytics):** IIT Bombay SEIL Residential Energy Dataset — real Indian residential smart-meter research data (per-apartment interval readings, `TS,V1,V2,V3,W1,W2,W3`). Place `1.csv` … `39.csv` (or the official `virtual_dataset.csv`) in `data/raw/indian_smart_meter/`, then run:
```cmd
python data/download_indian_dataset.py
```
This is described honestly as Indian residential smart-meter *research* data — never as nationwide or Tamil Nadu household data.

**Official context (not ML training data):** Central Electricity Authority (CEA) national statistics, kept in `data/official/cea_2025/`, used only for the separate `/api/benchmark` endpoint. FY 2024-25 values are explicitly labeled provisional.

## Machine Learning

- **Forecasting:** RandomForestRegressor with lag + calendar features vs. a persistence baseline; the better-performing model on a chronological holdout is reported.
- **Anomaly detection:** rolling z-score + Isolation Forest, cross-validated against each other.
- **Clustering:** K-Means with StandardScaler, k chosen by silhouette score.

All metrics shown in the UI come from an actual model run on the active dataset. If there isn't enough data, the UI says so instead of inventing a number.

## Novelty / Differentiation

This project does not claim to invent forecasting, anomaly detection, or clustering for electricity data — these are established techniques with substantial prior research. The system's differentiation lies in integrating established Data Science techniques into an explainable, scenario-driven, India-focused energy decision-support workflow — connecting analysis (EDA, forecasting, anomalies, clustering) directly to explainable findings (Consumption DNA, Health Score) and then to actionable, simulated interventions (What-If, recommendation-linked scenarios).

## Installation

```cmd
cd backend
py -3.12 -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run

Backend:
```cmd
cd backend
venv\Scripts\activate
python -m uvicorn app.main:app --reload
```
Verify: `http://127.0.0.1:8000/api/health`, `http://127.0.0.1:8000/api/benchmark`, `http://127.0.0.1:8000/docs`

Frontend (second terminal):
```cmd
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173`. Production build check: `npm run build`.

## Data Setup

Place the Indian dataset files as described above, then run `python data/download_indian_dataset.py` to verify them before starting the backend.

## Limitations (honest)

- Forecast uncertainty is derived from backtest residual spread (a reasonable, defensible proxy), not a formally calibrated prediction interval — labeled as "expected range," not a statistical confidence interval.
- Clustering and building-priority features require a dataset with a `meter_id`/`building` column; a plain single-series upload won't populate them (by design — no fabricated hierarchy).
- No PDF/Excel report generation beyond browser print-to-PDF.
- This was developed and syntax/logic-verified in a sandboxed environment without the ability to run `npm install` / `pip install` / boot a live server end-to-end — run the commands above on your machine as the real verification.
