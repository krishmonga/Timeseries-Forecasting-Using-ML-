from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.ts_forecasting.advanced_models import run_advanced_models
from src.ts_forecasting.data import clean_series, load_dataset
from src.ts_forecasting.deep_models import run_deep_learning_models
from src.ts_forecasting.evaluation import mape
from src.ts_forecasting.models import (
    ar_forecast,
    arima_forecast,
    holt_winters_forecast,
    ma_forecast,
    moving_average_forecast,
    naive_forecast,
    run_ml_models,
    sarima_forecast,
)


def safe_path(path: Path) -> Path:
    try:
        with path.open("a", encoding="utf-8"):
            pass
        return path
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def evaluate_against_actuals(pred: pd.Series, actual: pd.Series) -> Dict[str, float]:
    merged = pd.concat([actual.rename("actual"), pred.rename("pred")], axis=1).dropna()
    if merged.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}
    err = merged["actual"] - merged["pred"]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape_value = float(mape(merged["actual"].values, merged["pred"].values))
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape_value}


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast Walmart weekly sales for year 2026.")
    parser.add_argument(
        "--actual-2026-csv",
        type=str,
        default="",
        help="Optional CSV with columns Date and Weekly_Sales for evaluation.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    series_raw, info = load_dataset(project_root)
    series_clean, _ = clean_series(series_raw)

    forecast_index = pd.date_range("2026-01-02", "2026-12-25", freq="W-FRI")
    horizon = len(forecast_index)
    predictions: Dict[str, np.ndarray] = {}

    # Basic + statistical
    predictions["Naive"] = naive_forecast(series_clean, horizon).predictions
    predictions["MovingAverage"] = moving_average_forecast(series_clean, horizon, window=4).predictions
    predictions["AR"] = ar_forecast(series_clean, horizon, p=5).predictions
    predictions["MA"] = ma_forecast(series_clean, horizon, q=3).predictions
    predictions["ARIMA"] = arima_forecast(series_clean, horizon, order=(3, 1, 3)).predictions
    predictions["SARIMA"] = sarima_forecast(
        series_clean, horizon, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)
    ).predictions
    predictions["HoltWinters"] = holt_winters_forecast(series_clean, horizon, seasonal_periods=52).predictions

    # ML models
    lags = [1, 2, 3, 4, 8, 12]
    rolling_windows = [4, 8, 12]
    ml_outputs = run_ml_models(series_clean, forecast_index, lags, rolling_windows)
    for name, out in ml_outputs.items():
        predictions[name] = out.predictions

    # Deep + advanced
    deep_outputs = run_deep_learning_models(series_clean, forecast_index, lookback=12)
    for name, out in deep_outputs.items():
        predictions[name] = out.predictions
    adv_outputs = run_advanced_models(series_clean, forecast_index, lags, rolling_windows)
    for name, out in adv_outputs.items():
        predictions[name] = out.predictions

    all_df = pd.DataFrame(index=forecast_index)
    all_df.index.name = "Date"
    for model_name, pred in predictions.items():
        all_df[f"{model_name}_Pred"] = pred.astype(float)
    all_path = safe_path(output_dir / "forecast_2026_all_models.csv")
    all_df.to_csv(all_path)

    # Pick best model from latest comparison if available.
    best_model = "LinearRegression"
    candidates = sorted(output_dir.glob("model_comparison*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        comp = pd.read_csv(candidates[0])
        comp = comp.dropna(subset=["RMSE"])
        if not comp.empty:
            best_model = str(comp.iloc[0]["model"])
    best_pred = all_df[f"{best_model}_Pred"]
    best_df = pd.DataFrame(
        {
            "Date": forecast_index,
            "Best_Model": best_model,
            "Predicted_Weekly_Sales": best_pred.values,
        }
    )
    best_path = safe_path(output_dir / "forecast_2026_best_model.csv")
    best_df.to_csv(best_path, index=False)

    print(f"Dataset: {info.name}")
    print(f"Forecast horizon points: {horizon}")
    print(f"Best model selected from comparison: {best_model}")
    print(f"All models forecast file: {all_path}")
    print(f"Best-model forecast file: {best_path}")

    if args.actual_2026_csv:
        actual_df = pd.read_csv(args.actual_2026_csv, parse_dates=["Date"])
        actual_series = (
            actual_df.groupby("Date", as_index=True)["Weekly_Sales"].sum().sort_index().reindex(forecast_index)
        )
        metrics = evaluate_against_actuals(best_pred, actual_series)
        eval_df = pd.DataFrame([{"model": best_model, **metrics}])
        eval_path = safe_path(output_dir / "forecast_2026_evaluation.csv")
        eval_df.to_csv(eval_path, index=False)
        print(f"2026 evaluation file: {eval_path}")


if __name__ == "__main__":
    main()

