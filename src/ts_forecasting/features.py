from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def train_test_split_time(
    series: pd.Series, test_size: float = 0.2
) -> Tuple[pd.Series, pd.Series]:
    split_idx = int(len(series) * (1 - test_size))
    train = series.iloc[:split_idx].copy()
    test = series.iloc[split_idx:].copy()
    return train, test


def build_supervised_frame(
    series: pd.Series,
    lags: Sequence[int],
    rolling_windows: Sequence[int],
) -> pd.DataFrame:
    frame = pd.DataFrame(index=series.index)
    frame["y"] = series.values

    for lag in lags:
        frame[f"lag_{lag}"] = series.shift(lag)

    for window in rolling_windows:
        frame[f"roll_mean_{window}"] = series.shift(1).rolling(window).mean()
        frame[f"roll_std_{window}"] = series.shift(1).rolling(window).std()

    frame["month"] = frame.index.month
    frame["quarter"] = frame.index.quarter
    frame["weekofyear"] = frame.index.isocalendar().week.astype(int)
    frame["year"] = frame.index.year
    return frame.dropna()


def build_feature_row(
    history: List[float],
    timestamp: pd.Timestamp,
    lags: Sequence[int],
    rolling_windows: Sequence[int],
) -> Dict[str, float]:
    row: Dict[str, float] = {}
    hist = np.array(history, dtype=float)

    for lag in lags:
        row[f"lag_{lag}"] = float(hist[-lag])

    for window in rolling_windows:
        window_vals = hist[-window:]
        row[f"roll_mean_{window}"] = float(np.mean(window_vals))
        row[f"roll_std_{window}"] = float(np.std(window_vals))

    row["month"] = float(timestamp.month)
    row["quarter"] = float(timestamp.quarter)
    row["weekofyear"] = float(timestamp.isocalendar().week)
    row["year"] = float(timestamp.year)
    return row


def recursive_ml_forecast(
    model,
    train_series: pd.Series,
    horizon_index: Iterable[pd.Timestamp],
    lags: Sequence[int],
    rolling_windows: Sequence[int],
) -> np.ndarray:
    history = train_series.astype(float).tolist()
    preds: List[float] = []

    for ts in horizon_index:
        row = build_feature_row(history, pd.Timestamp(ts), lags, rolling_windows)
        x_next = pd.DataFrame([row], columns=list(row.keys()))
        pred = float(model.predict(x_next)[0])
        preds.append(pred)
        history.append(pred)
    return np.array(preds)

