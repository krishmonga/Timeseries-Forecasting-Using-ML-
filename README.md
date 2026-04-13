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

## 5) Production Pipeline (2026+ Ready)

### Train or retrain with optional new data

```bash
python run_production_pipeline.py --retrain --horizons "30,60,90"
```

### Use existing saved models only

```bash
python run_production_pipeline.py --use-existing-model --horizons "30,60,90"
```

### Append new rows and retrain

```bash
python run_production_pipeline.py --append-csv "path\to\new_rows.csv" --retrain
```

Production outputs are written to `outputs/production/`:

- `actual_vs_predicted.csv`
- `model_comparison_production.csv`
- `future_predictions_all_models.csv`
- `future_predictions_best_model.csv`
- `metadata.json` (diagnostics + retraining policy + selected model)

### Streamlit App

```bash
streamlit run streamlit_app.py
```

App features:

- Upload full dataset or append rows
- Choose `Use existing trained model` or `Retrain model`
- Predict next 30/60/90 days
- View model comparison + future predictions instantly

## 6) Outputs

After running, check:

- `outputs/model_comparison.csv` → final ranked model table
- `outputs/run_report.json` → dataset + stationarity + quality reports
- `outputs/plots/01_raw_series.png`
- `outputs/plots/02_cleaned_series.png`
- `outputs/plots/03_model_forecasts.png`

## 7) How to Interpret Model Choice

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

## 8) Hyperparameter Tuning

- XGBoost uses `RandomizedSearchCV` with `TimeSeriesSplit`.
- LSTM tuning searches combinations of lookback, epochs, units, and layer depth.
- ARIMA/SARIMA are tuned using `auto_arima`.
- Holt-Winters seasonal period is selected from candidates and evaluated.

## 9) Production Notes

- Keep time-based split strict (never random split for forecasting).
- Retrain on rolling windows for non-stationary behavior.
- Track metrics by horizon (1-step, 4-step, 12-step) for better reliability checks.
- Add experiment tracking (MLflow/W&B) if this moves to team production.
- The pipeline stores model artifacts and can reload them without retraining.
- Best model is selected automatically by lowest MAPE, then RMSE.
- If both LSTM and XGBoost are strong, an ensemble (`0.6*LSTM + 0.4*XGB`) is evaluated.

# Timeseries-Forecasting-Using-ML-
