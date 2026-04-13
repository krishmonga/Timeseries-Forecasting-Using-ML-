from __future__ import annotations

import json
from pathlib import Path
from typing import Dict
from datetime import datetime

import numpy as np

from .advanced_models import run_advanced_models
from .data import adf_check, clean_series, load_dataset, plot_series
from .deep_models import run_deep_learning_models
from .evaluation import evaluate_models, plot_forecasts
from .features import train_test_split_time
from .models import (
    ar_forecast,
    arima_forecast,
    holt_winters_forecast,
    ma_forecast,
    moving_average_forecast,
    naive_forecast,
    run_ml_models,
    sarima_forecast,
)


def _safe_output_path(path: Path) -> Path:
    try:
        # Fast check for lock/write availability.
        with path.open("a", encoding="utf-8"):
            pass
        return path
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def run_pipeline(project_root: Path) -> None:
    output_dir = project_root / "outputs"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    series_raw, info = load_dataset(project_root)
    series_clean, quality_report = clean_series(series_raw)

    plot_series(series_raw, "Raw Time Series", plots_dir / "01_raw_series.png")
    plot_series(series_clean, "Cleaned Time Series", plots_dir / "02_cleaned_series.png")

    adf_result = adf_check(series_clean)
    train, test = train_test_split_time(series_clean, test_size=0.2)
    horizon = len(test)

    predictions: Dict[str, np.ndarray] = {}
    notes: Dict[str, str] = {}

    def register(name: str, output) -> None:
        predictions[name] = output.predictions.astype(float)
        notes[name] = output.notes

    register("Naive", naive_forecast(train, horizon))
    register("MovingAverage", moving_average_forecast(train, horizon, window=4))

    for name, fn in [
        ("AR", lambda: ar_forecast(train, horizon, p=5)),
        ("MA", lambda: ma_forecast(train, horizon, q=3)),
        ("ARIMA", lambda: arima_forecast(train, horizon, order=(3, 1, 3))),
        ("SARIMA", lambda: sarima_forecast(train, horizon, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))),
        ("HoltWinters", lambda: holt_winters_forecast(train, horizon, seasonal_periods=52)),
    ]:
        try:
            register(name, fn())
        except Exception as exc:
            predictions[name] = np.full(horizon, np.nan)
            notes[name] = f"failed ({exc})"

    lags = [1, 2, 3, 4, 8, 12]
    rolling_windows = [4, 8, 12]

    ml_outputs = run_ml_models(train, test.index, lags, rolling_windows)
    for model_name, output in ml_outputs.items():
        register(model_name, output)

    deep_outputs = run_deep_learning_models(train, test.index, lookback=12)
    for model_name, output in deep_outputs.items():
        register(model_name, output)

    advanced_outputs = run_advanced_models(train, test.index, lags, rolling_windows)
    for model_name, output in advanced_outputs.items():
        register(model_name, output)

    comparison = evaluate_models(test, predictions, notes)
    comparison_csv_path = _safe_output_path(output_dir / "model_comparison.csv")
    comparison.to_csv(comparison_csv_path, index=False)
    # Date-wise prediction table so users can inspect each model vs actual sales.
    predictions_by_date = test.to_frame(name="Actual_Sales")
    for model_name, pred in predictions.items():
        predictions_by_date[f"{model_name}_Pred"] = pred
    predictions_by_date.index.name = "Date"
    predictions_by_date_csv_path = _safe_output_path(output_dir / "predictions_by_date.csv")
    predictions_by_date.to_csv(predictions_by_date_csv_path)

    # Explicit "next week" predictions (first step in the test horizon).
    next_week_date = test.index[0]
    next_week_actual = float(test.iloc[0])
    next_week_rows = []
    for model_name, pred in predictions.items():
        next_pred = float(pred[0]) if len(pred) > 0 else np.nan
        next_week_rows.append(
            {
                "model": model_name,
                "next_week_date": str(next_week_date.date()),
                "next_week_actual_sales": next_week_actual,
                "next_week_predicted_sales": next_pred,
                "next_week_abs_error": abs(next_week_actual - next_pred) if not np.isnan(next_pred) else np.nan,
                "notes": notes.get(model_name, ""),
            }
        )
    import pandas as pd

    next_week_df = pd.DataFrame(next_week_rows).sort_values("next_week_abs_error", na_position="last")
    next_week_predictions_csv_path = _safe_output_path(output_dir / "next_week_predictions.csv")
    next_week_df.to_csv(next_week_predictions_csv_path, index=False)
    plot_forecasts(train, test, predictions, plots_dir / "03_model_forecasts.png")

    report = {
        "dataset": {
            "name": info.name,
            "source_path": info.source_path,
            "records": info.records,
            "series_points": len(series_raw),
            "train_points": len(train),
            "test_points": len(test),
        },
        "quality_report": quality_report,
        "adf_result": adf_result,
        "artifacts": {
            "comparison_csv": str(comparison_csv_path),
            "predictions_by_date_csv": str(predictions_by_date_csv_path),
            "next_week_predictions_csv": str(next_week_predictions_csv_path),
            "plots": [
                str(plots_dir / "01_raw_series.png"),
                str(plots_dir / "02_cleaned_series.png"),
                str(plots_dir / "03_model_forecasts.png"),
            ],
        },
    }
    run_report_path = _safe_output_path(output_dir / "run_report.json")
    with run_report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Pipeline complete.")
    print(f"Dataset: {info.name}")
    print(f"Comparison table: {comparison_csv_path}")
    print(f"Predictions by date: {predictions_by_date_csv_path}")
    print(f"Next week predictions: {next_week_predictions_csv_path}")
    print(f"Run report: {run_report_path}")

