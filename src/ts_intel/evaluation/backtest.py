import pandas as pd
from ts_intel.metrics import forecast_metrics
from ts_intel.models.baselines import persistence, seasonal_naive

def rolling_origin_backtest(series: pd.Series, horizon=1, initial_train_size=None,
                             step=1, model="seasonal_naive", season_length=7):
    if horizon < 1 or step < 1: raise ValueError("horizon and step must be positive")
    initial = initial_train_size or max(3, len(series)//2)
    if initial < 2 or initial+horizon > len(series):
        raise ValueError("Not enough data for initial training window plus horizon")
    records = []
    for origin in range(initial, len(series)-horizon+1, step):
        train = series.iloc[:origin]
        actual = series.iloc[origin:origin+horizon].to_numpy(dtype=float)
        if model == "persistence":
            pred = persistence(train, horizon)
        elif model == "seasonal_naive":
            pred = seasonal_naive(train, horizon, min(season_length, len(train)))
        else:
            raise ValueError("model must be persistence or seasonal_naive")
        records.append({"origin": origin, **forecast_metrics(actual, pred)})
    if not records: raise ValueError("No backtest folds generated")
    return pd.DataFrame(records)
