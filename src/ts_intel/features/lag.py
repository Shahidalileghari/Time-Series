import pandas as pd

def make_lag_features(series: pd.Series, lags=(1, 2, 3, 7), windows=(3, 7)) -> pd.DataFrame:
    if not lags or any(lag < 1 for lag in lags):
        raise ValueError("lags must contain positive integers")
    frame = pd.DataFrame(index=series.index)
    for lag in lags:
        frame[f"lag_{lag}"] = series.shift(lag)
    for window in windows:
        if window < 1:
            raise ValueError("rolling windows must be positive")
        frame[f"rolling_mean_{window}"] = series.shift(1).rolling(window).mean()
        frame[f"rolling_std_{window}"] = series.shift(1).rolling(window).std()
    return frame

def add_calendar_features(frame: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    result = frame.copy()
    ts = pd.to_datetime(result[timestamp_column], utc=True)
    result["day_of_week"] = ts.dt.dayofweek
    result["day_of_month"] = ts.dt.day
    result["month"] = ts.dt.month
    result["quarter"] = ts.dt.quarter
    return result
