import pandas as pd

def validate_series_frame(frame: pd.DataFrame, timestamp: str, target: str) -> pd.DataFrame:
    if timestamp not in frame or target not in frame:
        raise ValueError(f"Expected columns: {timestamp!r} and {target!r}")
    result = frame[[timestamp, target]].copy()
    result[timestamp] = pd.to_datetime(result[timestamp], errors="coerce", utc=True)
    result[target] = pd.to_numeric(result[target], errors="coerce")
    if result[timestamp].isna().any():
        raise ValueError("Timestamp column contains invalid or missing values")
    if result[target].isna().any():
        raise ValueError("Target column contains missing or non-numeric values")
    result = result.sort_values(timestamp).reset_index(drop=True)
    if result[timestamp].duplicated().any():
        raise ValueError("Duplicate timestamps found; aggregate or resolve them explicitly")
    if len(result) < 3:
        raise ValueError("At least three observations are required")
    return result
