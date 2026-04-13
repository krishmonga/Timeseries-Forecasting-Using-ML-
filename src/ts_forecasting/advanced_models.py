from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

from .features import build_supervised_frame, recursive_ml_forecast
from .models import ModelOutput


def prophet_forecast(train: pd.Series, test_index: pd.DatetimeIndex) -> ModelOutput:
    try:
        from prophet import Prophet  # type: ignore

        prophet_df = train.reset_index()
        prophet_df.columns = ["ds", "y"]
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        model.fit(prophet_df)

        future = pd.DataFrame({"ds": test_index})
        forecast = model.predict(future)
        return ModelOutput(
            predictions=forecast["yhat"].values,
            notes="Prophet with yearly seasonality",
        )
    except Exception as exc:  # pragma: no cover
        return ModelOutput(
            predictions=np.full(len(test_index), np.nan),
            notes=f"skipped ({exc})",
        )


def hybrid_ml_arima_forecast(
    train: pd.Series,
    test_index: pd.DatetimeIndex,
    lags: Sequence[int],
    rolling_windows: Sequence[int],
) -> ModelOutput:
    frame = build_supervised_frame(train, lags, rolling_windows)
    x_train = frame.drop(columns=["y"])
    y_train = frame["y"]

    # Base learner for nonlinear dynamics.
    base_model = LinearRegression()
    base_model.fit(x_train, y_train)
    train_pred = base_model.predict(x_train)
    residuals = y_train - train_pred

    residual_model = ARIMA(residuals, order=(1, 0, 1)).fit()
    base_forecast = recursive_ml_forecast(base_model, train, test_index, lags, rolling_windows)
    residual_forecast = residual_model.forecast(steps=len(test_index)).values
    combined = base_forecast + residual_forecast
    return ModelOutput(
        predictions=combined,
        notes="LinearRegression + ARIMA(1,0,1) on residuals",
    )


def run_advanced_models(
    train: pd.Series,
    test_index: pd.DatetimeIndex,
    lags: Sequence[int],
    rolling_windows: Sequence[int],
) -> Dict[str, ModelOutput]:
    outputs = {
        "Prophet": prophet_forecast(train, test_index),
    }
    try:
        outputs["Hybrid_ML_ARIMA"] = hybrid_ml_arima_forecast(
            train,
            test_index,
            lags,
            rolling_windows,
        )
    except Exception as exc:  # pragma: no cover
        outputs["Hybrid_ML_ARIMA"] = ModelOutput(
            predictions=np.full(len(test_index), np.nan),
            notes=f"skipped ({exc})",
        )
    return outputs

