import pandas as pd

def chronological_split(series: pd.Series, train_size: int):
    if not 1 <= train_size < len(series):
        raise ValueError("train_size must be between 1 and len(series)-1")
    return series.iloc[:train_size].copy(), series.iloc[train_size:].copy()
