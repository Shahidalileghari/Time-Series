import pandas as pd
import typer
from rich import print
from ts_intel.data.validation import validate_series_frame
from ts_intel.evaluation.backtest import rolling_origin_backtest

app = typer.Typer(help="Time Series Intelligence CLI")

@app.command()
def evaluate(csv: str, timestamp: str="timestamp", target: str="value",
             horizon: int=7, initial_train_size: int=90,
             model: str="seasonal_naive", season_length: int=7):
    """Evaluate a baseline with expanding-window rolling-origin backtesting."""
    frame = validate_series_frame(pd.read_csv(csv), timestamp, target)
    series = frame.set_index(timestamp)[target]
    result = rolling_origin_backtest(series, horizon, initial_train_size,
                                     model=model, season_length=season_length)
    print(result.to_string(index=False))
    print("\nMean metrics:")
    print(result[["mae","rmse","smape"]].mean().to_dict())

if __name__ == "__main__":
    app()
