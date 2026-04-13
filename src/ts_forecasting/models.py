from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .features import build_supervised_frame, recursive_ml_forecast


@dataclass
class ModelOutput:
    predictions: np.ndarray
    notes: str = ""


def naive_forecast(train: pd.Series, horizon: int) -> ModelOutput:
    return ModelOutput(predictions=np.repeat(float(train.iloc[-1]), horizon))


def moving_average_forecast(train: pd.Series, horizon: int, window: int = 4) -> ModelOutput:
    history = train.astype(float).tolist()
    preds = []
    for _ in range(horizon):
        preds.append(float(np.mean(history[-window:])))
        history.append(preds[-1])
    return ModelOutput(predictions=np.array(preds), notes=f"window={window}")


def ar_forecast(train: pd.Series, horizon: int, p: int = 5) -> ModelOutput:
    fit = ARIMA(train, order=(p, 0, 0)).fit()
    return ModelOutput(predictions=fit.forecast(steps=horizon).values, notes=f"p={p}")


def ma_forecast(train: pd.Series, horizon: int, q: int = 3) -> ModelOutput:
    fit = ARIMA(train, order=(0, 0, q)).fit()
    return ModelOutput(predictions=fit.forecast(steps=horizon).values, notes=f"q={q}")


def arima_forecast(train: pd.Series, horizon: int, order: Tuple[int, int, int] = (3, 1, 3)) -> ModelOutput:
    fit = ARIMA(train, order=order).fit()
    return ModelOutput(predictions=fit.forecast(steps=horizon).values, notes=f"order={order}")


def sarima_forecast(
    train: pd.Series,
    horizon: int,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 52),
) -> ModelOutput:
    fit = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    return ModelOutput(
        predictions=fit.forecast(steps=horizon).values,
        notes=f"order={order}, seasonal_order={seasonal_order}",
    )


def holt_winters_forecast(train: pd.Series, horizon: int, seasonal_periods: int = 52) -> ModelOutput:
    fit = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=seasonal_periods,
    ).fit(optimized=True)
    return ModelOutput(
        predictions=fit.forecast(horizon).values,
        notes=f"seasonal_periods={seasonal_periods}",
    )


def run_ml_models(
    train: pd.Series,
    test_index: pd.DatetimeIndex,
    lags: Sequence[int],
    rolling_windows: Sequence[int],
) -> Dict[str, ModelOutput]:
    frame = build_supervised_frame(train, lags, rolling_windows)
    x_train = frame.drop(columns=["y"])
    y_train = frame["y"]
    outputs: Dict[str, ModelOutput] = {}

    linear = LinearRegression()
    linear.fit(x_train, y_train)
    outputs["LinearRegression"] = ModelOutput(
        predictions=recursive_ml_forecast(linear, train, test_index, lags, rolling_windows),
    )

    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
    cv_splits = min(4, max(2, len(x_train) // 25))
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    grid = GridSearchCV(
        rf_base,
        param_grid={
            "n_estimators": [200, 400],
            "max_depth": [4, 8, None],
            "min_samples_leaf": [1, 3, 5],
        },
        scoring="neg_mean_absolute_error",
        cv=tscv,
        n_jobs=-1,
    )
    grid.fit(x_train, y_train)
    best_rf = grid.best_estimator_
    outputs["RandomForest"] = ModelOutput(
        predictions=recursive_ml_forecast(best_rf, train, test_index, lags, rolling_windows),
        notes=f"best_params={grid.best_params_}",
    )

    try:
        from xgboost import XGBRegressor  # type: ignore

        xgb = XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
        xgb.fit(x_train, y_train)
        outputs["XGBoost"] = ModelOutput(
            predictions=recursive_ml_forecast(xgb, train, test_index, lags, rolling_windows),
            notes="default tuned config",
        )
    except Exception as exc:  # pragma: no cover
        outputs["XGBoost"] = ModelOutput(
            predictions=np.full(len(test_index), np.nan),
            notes=f"skipped ({exc})",
        )

    return outputs

