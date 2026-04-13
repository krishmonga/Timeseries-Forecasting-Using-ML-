from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .models import ModelOutput


def _make_sequences(values: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(lookback, len(values)):
        x.append(values[i - lookback : i])
        y.append(values[i])
    return np.array(x), np.array(y)


def run_deep_learning_models(
    train: pd.Series,
    test_index: pd.DatetimeIndex,
    lookback: int = 12,
) -> Dict[str, ModelOutput]:
    outputs: Dict[str, ModelOutput] = {}
    try:
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
        from tensorflow.keras.layers import GRU, LSTM, Dense  # type: ignore
        from tensorflow.keras.models import Sequential  # type: ignore

        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train.values.reshape(-1, 1)).flatten()
        x_train, y_train = _make_sequences(train_scaled, lookback)
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

        def build_and_forecast(cell_type: str) -> np.ndarray:
            model = Sequential()
            if cell_type == "LSTM":
                model.add(LSTM(32, activation="tanh", input_shape=(lookback, 1)))
            else:
                model.add(GRU(32, activation="tanh", input_shape=(lookback, 1)))
            model.add(Dense(1))
            model.compile(optimizer="adam", loss="mse")
            es = EarlyStopping(patience=5, restore_best_weights=True)
            model.fit(
                x_train,
                y_train,
                epochs=40,
                batch_size=16,
                verbose=0,
                validation_split=0.2,
                callbacks=[es],
            )

            history = train_scaled.tolist()
            preds = []
            for _ in range(len(test_index)):
                x_next = np.array(history[-lookback:]).reshape(1, lookback, 1)
                pred_scaled = float(model.predict(x_next, verbose=0)[0][0])
                preds.append(pred_scaled)
                history.append(pred_scaled)
            preds = np.array(preds).reshape(-1, 1)
            return scaler.inverse_transform(preds).flatten()

        outputs["LSTM"] = ModelOutput(
            predictions=build_and_forecast("LSTM"),
            notes=f"lookback={lookback}, epochs<=40",
        )
        outputs["GRU"] = ModelOutput(
            predictions=build_and_forecast("GRU"),
            notes=f"lookback={lookback}, epochs<=40",
        )
    except Exception as exc:  # pragma: no cover
        outputs["LSTM"] = ModelOutput(
            predictions=np.full(len(test_index), np.nan),
            notes=f"skipped ({exc})",
        )
        outputs["GRU"] = ModelOutput(
            predictions=np.full(len(test_index), np.nan),
            notes=f"skipped ({exc})",
        )
    return outputs

