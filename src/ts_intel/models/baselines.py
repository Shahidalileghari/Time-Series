import numpy as np
import pandas as pd

def persistence(series: pd.Series, horizon: int) -> np.ndarray:
    _validate_horizon(horizon)
    if len(series) == 0: raise ValueError("series cannot be empty")
    return np.repeat(float(series.iloc[-1]), horizon)

def seasonal_naive(series: pd.Series, horizon: int, season_length: int) -> np.ndarray:
    _validate_horizon(horizon)
    if season_length < 1 or len(series) < season_length:
        raise ValueError("series must contain at least one complete season")
    pattern = np.asarray(series.iloc[-season_length:], dtype=float)
    return np.resize(pattern, horizon)

def autoregressive_forecast(series: pd.Series, horizon: int, lags=5) -> np.ndarray:
    _validate_horizon(horizon)
    if lags < 1 or len(series) <= lags:
        raise ValueError("Not enough observations for requested lags")
    values = np.asarray(series, dtype=float)
    x, y = [], []
    for i in range(lags, len(values)):
        x.append(values[i-lags:i][::-1]); y.append(values[i])
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0).fit(np.asarray(x), np.asarray(y))
    history = list(values)
    predictions = []
    for _ in range(horizon):
        row = np.asarray(history[-lags:][::-1]).reshape(1, -1)
        pred = float(model.predict(row)[0])
        predictions.append(pred); history.append(pred)
    return np.asarray(predictions)

def _validate_horizon(horizon):
    if horizon < 1: raise ValueError("horizon must be >= 1")
