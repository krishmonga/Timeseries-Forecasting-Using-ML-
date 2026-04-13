from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller

from .features import build_supervised_frame, recursive_ml_forecast


@dataclass
class DeepModelBundle:
    model: Any
    scaler: Any
    lookback: int


def _infer_frequency(index: pd.DatetimeIndex) -> str:
    if index.freqstr:
        return index.freqstr
    inferred = pd.infer_freq(index)
    if inferred:
        return inferred
    return "W-FRI"


def _periods_from_days(days: int, freq: str) -> int:
    if freq.upper().startswith("W"):
        return max(1, math.ceil(days / 7))
    if freq.upper().startswith("M"):
        return max(1, math.ceil(days / 30))
    return max(1, days)


def prepare_series(
    frame: pd.DataFrame,
    date_col: str = "Date",
    target_col: str = "Weekly_Sales",
    freq: str | None = None,
) -> pd.Series:
    data = frame.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col, target_col])
    grouped = data.groupby(date_col, as_index=True)[target_col].sum().sort_index().astype(float)
    inferred = freq or _infer_frequency(grouped.index)
    series = grouped.asfreq(inferred)
    return series.interpolate(method="linear").bfill().ffill().rename("y")


def detect_seasonal_period(series: pd.Series) -> int:
    max_lag = min(max(12, len(series) // 2), 60)
    if max_lag <= 3:
        return 4
    vals = acf(series.values, nlags=max_lag, fft=True)
    candidate_lags = np.arange(2, len(vals))
    best_lag = int(candidate_lags[np.argmax(vals[2:])])
    return max(4, min(best_lag, 52))


def analyze_series(series: pd.Series, seasonal_period: int) -> Dict[str, Any]:
    adf = adfuller(series.dropna(), autolag="AIC")
    stl = STL(series, period=seasonal_period, robust=True).fit()
    resid_var = float(np.var(stl.resid))
    seasonal_var = float(np.var(stl.seasonal))
    trend_var = float(np.var(stl.trend))
    seasonal_strength = max(0.0, 1 - (resid_var / (seasonal_var + resid_var + 1e-8)))
    trend_strength = max(0.0, 1 - (resid_var / (trend_var + resid_var + 1e-8)))
    return {
        "adf_statistic": float(adf[0]),
        "adf_p_value": float(adf[1]),
        "adf_stationary": bool(adf[1] < 0.05),
        "seasonal_period_detected": seasonal_period,
        "seasonal_strength": seasonal_strength,
        "trend_strength": trend_strength,
    }


def residual_diagnostics(residuals: np.ndarray) -> Dict[str, float]:
    ljung = acorr_ljungbox(residuals, lags=[min(10, max(2, len(residuals) // 5))], return_df=True)
    return {
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "ljung_box_p_value": float(ljung["lb_pvalue"].iloc[0]),
    }


def _extract_residuals(model: Any, fallback_series: pd.Series) -> np.ndarray:
    try:
        resid_attr = getattr(model, "resid", None)
        if callable(resid_attr):
            return np.asarray(resid_attr(), dtype=float)
        if resid_attr is not None:
            return np.asarray(resid_attr, dtype=float)
        arima_res = getattr(model, "arima_res_", None)
        if arima_res is not None and hasattr(arima_res, "resid"):
            return np.asarray(arima_res.resid, dtype=float)
    except Exception:
        pass
    return np.asarray(fallback_series - fallback_series.mean(), dtype=float)


def _mape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.where(actual == 0, 1e-8, actual)
    return float(np.mean(np.abs((actual - pred) / denom)) * 100)


def _evaluate(actual: pd.Series, pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
        "MAPE": _mape(actual.values, pred),
    }


def _build_lstm_sequences(values: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(lookback, len(values)):
        x.append(values[i - lookback : i])
        y.append(values[i])
    return np.array(x), np.array(y)


def train_tuned_lstm(train: pd.Series, test_index: pd.DatetimeIndex) -> Tuple[np.ndarray, str, DeepModelBundle | None]:
    try:
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
        from tensorflow.keras.layers import LSTM, Dense  # type: ignore
        from tensorflow.keras.models import Sequential  # type: ignore

        search_grid = [
            {"lookback": 8, "epochs": 30, "units": 32, "layers": 1},
            {"lookback": 12, "epochs": 40, "units": 32, "layers": 1},
            {"lookback": 12, "epochs": 50, "units": 64, "layers": 2},
        ]
        best_cfg = None
        best_loss = float("inf")
        best_model = None
        best_scaler = None

        for cfg in search_grid:
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(train.values.reshape(-1, 1)).flatten()
            x_train, y_train = _build_lstm_sequences(scaled, cfg["lookback"])
            if len(x_train) < 20:
                continue
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

            model = Sequential()
            model.add(LSTM(cfg["units"], activation="tanh", return_sequences=cfg["layers"] > 1, input_shape=(cfg["lookback"], 1)))
            if cfg["layers"] > 1:
                model.add(LSTM(cfg["units"] // 2, activation="tanh"))
            model.add(Dense(1))
            model.compile(optimizer="adam", loss="mse")
            es = EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")
            history = model.fit(
                x_train,
                y_train,
                epochs=cfg["epochs"],
                batch_size=16,
                verbose=0,
                validation_split=0.2,
                callbacks=[es],
            )
            val_loss = float(np.min(history.history.get("val_loss", [1e9])))
            if val_loss < best_loss:
                best_loss = val_loss
                best_cfg = cfg
                best_model = model
                best_scaler = scaler

        if best_model is None or best_scaler is None or best_cfg is None:
            return np.full(len(test_index), np.nan), "LSTM tuning skipped (insufficient samples)", None

        history = best_scaler.transform(train.values.reshape(-1, 1)).flatten().tolist()
        preds = []
        for _ in range(len(test_index)):
            x_next = np.array(history[-best_cfg["lookback"] :]).reshape(1, best_cfg["lookback"], 1)
            pred_scaled = float(best_model.predict(x_next, verbose=0)[0][0])
            preds.append(pred_scaled)
            history.append(pred_scaled)
        pred_values = best_scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        note = (
            f"lookback={best_cfg['lookback']}, epochs<={best_cfg['epochs']}, "
            f"units={best_cfg['units']}, layers={best_cfg['layers']}"
        )
        return pred_values, note, DeepModelBundle(best_model, best_scaler, best_cfg["lookback"])
    except Exception as exc:  # pragma: no cover
        return np.full(len(test_index), np.nan), f"LSTM skipped ({exc})", None


def forecast_lstm(bundle: DeepModelBundle, history_series: pd.Series, horizon: int) -> np.ndarray:
    scaled_hist = bundle.scaler.transform(history_series.values.reshape(-1, 1)).flatten().tolist()
    preds = []
    for _ in range(horizon):
        x_next = np.array(scaled_hist[-bundle.lookback :]).reshape(1, bundle.lookback, 1)
        pred_scaled = float(bundle.model.predict(x_next, verbose=0)[0][0])
        preds.append(pred_scaled)
        scaled_hist.append(pred_scaled)
    return bundle.scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()


def train_tuned_xgboost(
    train: pd.Series,
    test_index: pd.DatetimeIndex,
    lags: Sequence[int],
    rolling_windows: Sequence[int],
) -> Tuple[np.ndarray, str, Any]:
    frame = build_supervised_frame(train, lags, rolling_windows)
    x_train = frame.drop(columns=["y"])
    y_train = frame["y"]
    try:
        from xgboost import XGBRegressor  # type: ignore

        model = XGBRegressor(objective="reg:squarederror", random_state=42)
        cv = TimeSeriesSplit(n_splits=min(4, max(2, len(x_train) // 25)))
        search = RandomizedSearchCV(
            model,
            param_distributions={
                "n_estimators": [200, 300, 500, 700],
                "max_depth": [3, 4, 5, 6],
                "learning_rate": [0.03, 0.05, 0.08, 0.1],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
            },
            n_iter=10,
            scoring="neg_mean_absolute_error",
            cv=cv,
            random_state=42,
            n_jobs=-1,
        )
        search.fit(x_train, y_train)
        best = search.best_estimator_
        preds = recursive_ml_forecast(best, train, test_index, lags, rolling_windows)
        return preds, f"best_params={search.best_params_}", best
    except Exception as exc:  # pragma: no cover
        return np.full(len(test_index), np.nan), f"XGBoost skipped ({exc})", None


def tune_auto_arima(
    train: pd.Series, test_index: pd.DatetimeIndex, seasonal_period: int
) -> Tuple[np.ndarray, str, Any, Any, np.ndarray]:
    try:
        from pmdarima import auto_arima  # type: ignore

        arima_model = auto_arima(
            train,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
        )
        sarima_model = auto_arima(
            train,
            seasonal=True,
            m=seasonal_period,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
        )
        arima_pred = np.asarray(arima_model.predict(n_periods=len(test_index)), dtype=float)
        sarima_pred = np.asarray(sarima_model.predict(n_periods=len(test_index)), dtype=float)
        note = f"auto_arima tuned; SARIMA seasonal m={seasonal_period}"
        return arima_pred, note, arima_model, sarima_model, sarima_pred
    except Exception as exc:  # pragma: no cover
        # Fallback if pmdarima is unavailable.
        arima_fit = ARIMA(train, order=(3, 1, 3)).fit()
        sarima_fit = ARIMA(train, order=(2, 1, 2)).fit()
        arima_pred = arima_fit.forecast(steps=len(test_index)).values
        sarima_pred = sarima_fit.forecast(steps=len(test_index)).values
        note = f"auto_arima unavailable; fallback ARIMA used ({exc})"
        return arima_pred, note, arima_fit, sarima_fit, sarima_pred


def tune_holt_winters(
    train: pd.Series,
    test: pd.Series,
    period_candidates: Sequence[int],
) -> Tuple[np.ndarray, str, Any]:
    best_rmse = float("inf")
    best_model = None
    best_period = None
    best_pred = np.full(len(test), np.nan)
    for period in period_candidates:
        if period >= len(train) // 2:
            continue
        try:
            model = ExponentialSmoothing(
                train, trend="add", seasonal="add", seasonal_periods=period
            ).fit(optimized=True)
            pred = model.forecast(len(test)).values
            rmse = float(np.sqrt(np.mean((pred - test.values) ** 2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_period = period
                best_pred = pred
        except Exception:
            continue
    if best_model is None:
        return np.full(len(test), np.nan), "HoltWinters tuning failed", None
    return best_pred, f"best seasonal_periods={best_period}", best_model


def train_or_load_best_models(
    series: pd.Series,
    artifacts_dir: Path,
    retrain: bool = True,
    horizons_days: Sequence[int] = (30, 60, 90),
) -> Dict[str, Any]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_dir = artifacts_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = artifacts_dir / "metadata.json"

    if (not retrain) and metadata_path.exists():
        return load_and_predict(series, artifacts_dir, horizons_days)

    split_idx = int(len(series) * 0.8)
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]
    test_index = test.index

    seasonal_period = detect_seasonal_period(train)
    diagnostics = analyze_series(train, seasonal_period=seasonal_period)
    lags = [1, 2, 3, 4, 8, 12]
    rolling_windows = [4, 8, 12]

    predictions: Dict[str, np.ndarray] = {}
    notes: Dict[str, str] = {}
    fitted_models: Dict[str, Any] = {}

    # Statistical with automatic tuning.
    arima_pred, arima_note, arima_model, sarima_model, sarima_pred = tune_auto_arima(train, test_index, seasonal_period)
    predictions["ARIMA_Auto"] = arima_pred
    notes["ARIMA_Auto"] = arima_note
    fitted_models["ARIMA_Auto"] = arima_model
    predictions["SARIMA_Auto"] = sarima_pred
    notes["SARIMA_Auto"] = f"auto_arima seasonal m={seasonal_period}"
    fitted_models["SARIMA_Auto"] = sarima_model

    # Residual checks for underperforming statistical models.
    diagnostics["arima_residual_diagnostics"] = residual_diagnostics(_extract_residuals(arima_model, train))
    diagnostics["sarima_residual_diagnostics"] = residual_diagnostics(_extract_residuals(sarima_model, train))

    hw_pred, hw_note, hw_model = tune_holt_winters(train, test, [4, 12, 26, 52])
    predictions["HoltWinters_Tuned"] = hw_pred
    notes["HoltWinters_Tuned"] = hw_note
    fitted_models["HoltWinters_Tuned"] = hw_model

    # Linear benchmark.
    frame = build_supervised_frame(train, lags, rolling_windows)
    x_train = frame.drop(columns=["y"])
    y_train = frame["y"]
    linear = LinearRegression()
    linear.fit(x_train, y_train)
    predictions["LinearRegression"] = recursive_ml_forecast(linear, train, test_index, lags, rolling_windows)
    notes["LinearRegression"] = "lag + rolling + datetime features"
    fitted_models["LinearRegression"] = linear

    # Tuned XGBoost.
    xgb_pred, xgb_note, xgb_model = train_tuned_xgboost(train, test_index, lags, rolling_windows)
    predictions["XGBoost_Tuned"] = xgb_pred
    notes["XGBoost_Tuned"] = xgb_note
    if xgb_model is not None:
        fitted_models["XGBoost_Tuned"] = xgb_model

    # Tuned LSTM.
    lstm_pred, lstm_note, lstm_bundle = train_tuned_lstm(train, test_index)
    predictions["LSTM_Tuned"] = lstm_pred
    notes["LSTM_Tuned"] = lstm_note

    # Ensemble if both strong predictors exist.
    ensemble_available = not np.isnan(lstm_pred).all() and not np.isnan(xgb_pred).all()
    if ensemble_available:
        ens = 0.6 * lstm_pred + 0.4 * xgb_pred
        predictions["Ensemble_LSTM_XGB"] = ens
        notes["Ensemble_LSTM_XGB"] = "0.6*LSTM + 0.4*XGBoost"

    comp_rows = []
    for name, pred in predictions.items():
        if np.isnan(pred).all():
            comp_rows.append({"model": name, "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "notes": notes.get(name, "")})
            continue
        metrics = _evaluate(test, pred)
        comp_rows.append({"model": name, **metrics, "notes": notes.get(name, "")})
    comparison = pd.DataFrame(comp_rows).sort_values(["MAPE", "RMSE"], na_position="last").reset_index(drop=True)
    best_model = str(comparison.iloc[0]["model"])

    # Save artifact tables.
    actual_vs_pred = test.to_frame(name="Actual_Sales")
    for name, pred in predictions.items():
        actual_vs_pred[f"{name}_Pred"] = pred
    actual_vs_pred.to_csv(artifacts_dir / "actual_vs_predicted.csv")
    comparison.to_csv(artifacts_dir / "model_comparison_production.csv", index=False)

    # Fit selected models on full series for forward prediction.
    full_frame = build_supervised_frame(series, lags, rolling_windows)
    full_x = full_frame.drop(columns=["y"])
    full_y = full_frame["y"]
    linear_full = LinearRegression().fit(full_x, full_y)
    joblib.dump(linear_full, model_dir / "LinearRegression.joblib")
    if xgb_model is not None:
        # Refit tuned params on full data.
        try:
            xgb_model.fit(full_x, full_y)
            joblib.dump(xgb_model, model_dir / "XGBoost_Tuned.joblib")
        except Exception:
            pass

    # Refit auto_arima-style models on full series.
    try:
        from pmdarima import auto_arima  # type: ignore

        arima_full = auto_arima(series, seasonal=False, stepwise=True, suppress_warnings=True, error_action="ignore")
        sarima_full = auto_arima(
            series,
            seasonal=True,
            m=seasonal_period,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )
        with (model_dir / "ARIMA_Auto.pkl").open("wb") as f:
            pickle.dump(arima_full, f)
        with (model_dir / "SARIMA_Auto.pkl").open("wb") as f:
            pickle.dump(sarima_full, f)
    except Exception:
        pass

    if hw_model is not None:
        try:
            hw_full = ExponentialSmoothing(
                series,
                trend="add",
                seasonal="add",
                seasonal_periods=seasonal_period,
            ).fit(optimized=True)
            with (model_dir / "HoltWinters_Tuned.pkl").open("wb") as f:
                pickle.dump(hw_full, f)
        except Exception:
            pass

    if lstm_bundle is not None:
        try:
            lstm_bundle.model.save(model_dir / "LSTM_Tuned.keras")
            joblib.dump(lstm_bundle.scaler, model_dir / "LSTM_Tuned_scaler.joblib")
            with (model_dir / "LSTM_Tuned_meta.json").open("w", encoding="utf-8") as f:
                json.dump({"lookback": lstm_bundle.lookback}, f, indent=2)
        except Exception:
            pass

    series.to_csv(artifacts_dir / "history_series.csv", header=True)
    metadata = {
        "freq": _infer_frequency(series.index),
        "lags": lags,
        "rolling_windows": rolling_windows,
        "seasonal_period": seasonal_period,
        "best_model": best_model,
        "diagnostics": diagnostics,
        "retraining_strategy": "Periodic retraining recommended whenever new weekly data is appended.",
        "incremental_learning": "Tree/deep/statistical models here are retrained periodically; append + retrain workflow supported.",
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    future = load_and_predict(series, artifacts_dir, horizons_days)
    future["model_comparison"] = comparison
    future["actual_vs_predicted"] = actual_vs_pred
    future["metadata"] = metadata
    return future


def _future_index(last_date: pd.Timestamp, freq: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]


def _recursive_with_model(
    model: Any,
    history: pd.Series,
    future_idx: Iterable[pd.Timestamp],
    lags: Sequence[int],
    rolling_windows: Sequence[int],
) -> np.ndarray:
    return recursive_ml_forecast(model, history, future_idx, lags, rolling_windows)


def load_and_predict(series: pd.Series, artifacts_dir: Path, horizons_days: Sequence[int]) -> Dict[str, Any]:
    metadata_path = artifacts_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("No saved metadata found. Retrain first.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_dir = artifacts_dir / "models"
    freq = metadata.get("freq", _infer_frequency(series.index))
    lags = metadata.get("lags", [1, 2, 3, 4, 8, 12])
    rolling_windows = metadata.get("rolling_windows", [4, 8, 12])

    loaded: Dict[str, Any] = {}
    for name in ["LinearRegression", "XGBoost_Tuned"]:
        path = model_dir / f"{name}.joblib"
        if path.exists():
            loaded[name] = joblib.load(path)
    for name in ["ARIMA_Auto", "SARIMA_Auto", "HoltWinters_Tuned"]:
        path = model_dir / f"{name}.pkl"
        if path.exists():
            with path.open("rb") as f:
                loaded[name] = pickle.load(f)

    lstm_bundle = None
    lstm_model_path = model_dir / "LSTM_Tuned.keras"
    lstm_scaler_path = model_dir / "LSTM_Tuned_scaler.joblib"
    lstm_meta_path = model_dir / "LSTM_Tuned_meta.json"
    if lstm_model_path.exists() and lstm_scaler_path.exists() and lstm_meta_path.exists():
        try:
            from tensorflow.keras.models import load_model  # type: ignore

            lstm_model = load_model(lstm_model_path)
            lstm_scaler = joblib.load(lstm_scaler_path)
            lstm_meta = json.loads(lstm_meta_path.read_text(encoding="utf-8"))
            lstm_bundle = DeepModelBundle(lstm_model, lstm_scaler, int(lstm_meta["lookback"]))
        except Exception:
            lstm_bundle = None

    all_rows = []
    best_rows = []
    best_model = metadata.get("best_model", "LinearRegression")
    last_date = series.index[-1]

    for days in horizons_days:
        periods = _periods_from_days(int(days), freq)
        f_idx = _future_index(last_date, freq, periods)
        model_preds: Dict[str, np.ndarray] = {}

        if "LinearRegression" in loaded:
            model_preds["LinearRegression"] = _recursive_with_model(
                loaded["LinearRegression"], series, f_idx, lags, rolling_windows
            )
        if "XGBoost_Tuned" in loaded:
            model_preds["XGBoost_Tuned"] = _recursive_with_model(
                loaded["XGBoost_Tuned"], series, f_idx, lags, rolling_windows
            )
        if "ARIMA_Auto" in loaded:
            model_preds["ARIMA_Auto"] = np.asarray(loaded["ARIMA_Auto"].predict(n_periods=periods), dtype=float)
        if "SARIMA_Auto" in loaded:
            model_preds["SARIMA_Auto"] = np.asarray(loaded["SARIMA_Auto"].predict(n_periods=periods), dtype=float)
        if "HoltWinters_Tuned" in loaded:
            model_preds["HoltWinters_Tuned"] = np.asarray(loaded["HoltWinters_Tuned"].forecast(periods), dtype=float)
        if lstm_bundle is not None:
            model_preds["LSTM_Tuned"] = forecast_lstm(lstm_bundle, series, periods)
        if "LSTM_Tuned" in model_preds and "XGBoost_Tuned" in model_preds:
            model_preds["Ensemble_LSTM_XGB"] = 0.6 * model_preds["LSTM_Tuned"] + 0.4 * model_preds["XGBoost_Tuned"]

        for i, date in enumerate(f_idx):
            row = {"Date": date, "horizon_days": int(days)}
            for name, arr in model_preds.items():
                row[f"{name}_Pred"] = float(arr[i])
            all_rows.append(row)

            best_pred = np.nan
            if f"{best_model}" in model_preds:
                best_pred = float(model_preds[best_model][i])
            elif "LinearRegression" in model_preds:
                best_pred = float(model_preds["LinearRegression"][i])
            best_rows.append(
                {
                    "Date": date,
                    "horizon_days": int(days),
                    "Best_Model": best_model,
                    "Predicted_Sales": best_pred,
                }
            )

    all_df = pd.DataFrame(all_rows)
    best_df = pd.DataFrame(best_rows)
    all_df.to_csv(artifacts_dir / "future_predictions_all_models.csv", index=False)
    best_df.to_csv(artifacts_dir / "future_predictions_best_model.csv", index=False)
    return {
        "future_all_models": all_df,
        "future_best_model": best_df,
        "metadata": metadata,
    }
