from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(y_true == 0, 1e-8, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def evaluate_models(
    y_true: pd.Series,
    predictions: Dict[str, np.ndarray],
    notes: Dict[str, str],
) -> pd.DataFrame:
    records = []
    actual = y_true.values.astype(float)
    for model_name, pred in predictions.items():
        if np.isnan(pred).all():
            records.append(
                {
                    "model": model_name,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "MAPE": np.nan,
                    "notes": notes.get(model_name, ""),
                }
            )
            continue
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        records.append(
            {
                "model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape(actual, pred),
                "notes": notes.get(model_name, ""),
            }
        )
    result = pd.DataFrame(records).sort_values("RMSE", na_position="last").reset_index(drop=True)
    return result


def plot_forecasts(
    train: pd.Series,
    test: pd.Series,
    predictions: Dict[str, np.ndarray],
    output_file: Path,
) -> None:
    plt.figure(figsize=(14, 6))
    plt.plot(train.index, train.values, label="Train", color="black")
    plt.plot(test.index, test.values, label="Test (Actual)", color="green")

    for model_name, pred in predictions.items():
        if np.isnan(pred).all():
            continue
        plt.plot(test.index, pred, label=model_name, alpha=0.8)

    plt.title("Forecast Comparison")
    plt.xlabel("Date")
    plt.ylabel("Target")
    plt.grid(alpha=0.2)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150)
    plt.close()

