# End-to-End Time Series Forecasting Project

This project is a complete forecasting pipeline built for learning and production-style experimentation.  
It starts from simple baselines and moves to statistical, ML, deep learning, and advanced hybrid models.

## 1) What This Project Covers

- Data loading (auto-loads a real dataset from `walmartseries/train.csv` if available)
- Data cleaning (missing values + outlier capping)
- Visualization (raw/cleaned series + multi-model forecast plot)
- Stationarity check (ADF test)
- Feature engineering (lags, rolling stats, datetime features)
- Time-based train/test split
- Model training and comparison:
  - **Basic:** Naive, Moving Average
  - **Statistical:** AR, MA, ARIMA, SARIMA, Holt-Winters
  - **Machine Learning:** Linear Regression, Random Forest, XGBoost
  - **Deep Learning:** LSTM, GRU
  - **Advanced:** Prophet, Hybrid (ML + ARIMA residuals)
- Evaluation with MAE, RMSE, MAPE
- Final comparison table and artifacts

## 2) Project Structure

```text
timeseries/
├─ walmartseries/
│  └─ train.csv
├─ src/
│  └─ ts_forecasting/
│     ├─ data.py
│     ├─ features.py
│     ├─ models.py
│     ├─ deep_models.py
│     ├─ advanced_models.py
│     ├─ evaluation.py
│     └─ pipeline.py
├─ run_pipeline.py
├─ requirements.txt
└─ outputs/                 # created after execution
```

## 3) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If heavy libraries (`tensorflow`, `prophet`, or `xgboost`) fail to install, the pipeline still runs and marks those models as skipped.

## 4) Run

```bash
python run_pipeline.py
```

### Forecast for 2026

```bash
python forecast_2026.py
```

This generates:

- `outputs/forecast_2026_all_models.csv` (all models weekly forecasts for 2026)
- `outputs/forecast_2026_best_model.csv` (best model only, selected from latest comparison)

If you have actual 2026 sales:

```bash
python forecast_2026.py --actual-2026-csv "path\to\actual_2026.csv"
```

Expected columns in actual CSV: `Date`, `Weekly_Sales`.

## 5) Outputs

After running, check:

- `outputs/model_comparison.csv` → final ranked model table
- `outputs/run_report.json` → dataset + stationarity + quality reports
- `outputs/plots/01_raw_series.png`
- `outputs/plots/02_cleaned_series.png`
- `outputs/plots/03_model_forecasts.png`

## 6) How to Interpret Model Choice

- **Naive / Moving Average**
  - Best when you need fast baselines and sanity checks
  - Weak for regime shifts and complex seasonality

- **AR / MA / ARIMA**
  - Strong for linear autocorrelation and short-memory patterns
  - Need stationarity assumptions and parameter tuning

- **SARIMA / Holt-Winters**
  - Good for recurring seasonality (weekly/monthly/yearly)
  - Can struggle with nonlinear effects or many external drivers

- **Linear Regression / Random Forest / XGBoost**
  - Great with engineered lag/time features and nonlinear patterns
  - Require careful leakage prevention and feature pipeline discipline

- **LSTM / GRU**
  - Useful for long sequential dependencies and large datasets
  - Heavier training cost and often less interpretable

- **Prophet**
  - Fast to deploy for business time series with trend/seasonality/holidays
  - Can underperform on very noisy or highly custom dynamics

- **Hybrid (ML + ARIMA residuals)**
  - Helpful when one model captures trend while residuals keep autocorrelation
  - More moving parts; harder to debug and maintain

## 7) Hyperparameter Tuning

- Random Forest uses `GridSearchCV` with `TimeSeriesSplit`.
- You can extend tuning with Optuna for XGBoost or deep models.

## 8) Production Notes

- Keep time-based split strict (never random split for forecasting).
- Retrain on rolling windows for non-stationary behavior.
- Track metrics by horizon (1-step, 4-step, 12-step) for better reliability checks.
- Add experiment tracking (MLflow/W&B) if this moves to team production.

# Timeseries-Forecasting-Using-ML-
