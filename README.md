# Time Series Intelligence

An advanced, reproducible time-series engineering lab covering forecasting, anomaly detection, change-point analysis, probabilistic prediction, temporal feature engineering, and rolling-origin evaluation.

This is a modular project scaffold with working classical ML baselines and extensible deep-learning interfaces—not a collection of beginner notebooks.

## Capabilities

- Time-aware data validation, sorting, resampling, missing-value handling, and leakage-safe splitting
- Lag, rolling-window, calendar, and difference features
- Baselines: persistence, seasonal naive, and autoregression
- Supervised forecasting with scikit-learn-compatible estimators
- Rolling-origin backtesting and MAE/RMSE/sMAPE metrics
- Residual-based anomaly scoring and configurable thresholding
- Change-point detection baseline using rolling distribution shifts
- Optional PyTorch LSTM forecaster
- CLI for training/evaluation on a CSV time series
- Unit tests, configuration examples, experiment template, Docker, and CI

## Quick start

Python 3.10–3.12 recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

Run:
```bash
pytest
ts-intel --help
```

Optional deep-learning support:
```bash
pip install -e ".[torch]"
```

## CSV input format

CSV must contain a timestamp column and a numeric target column. Example:

```csv
timestamp,value
2025-01-01,120.5
2025-01-02,121.1
2025-01-03,119.8
```

## CLI example

```bash
ts-intel evaluate   --csv data/example.csv   --timestamp timestamp   --target value   --horizon 7   --initial-train-size 90
```

The CLI reports a rolling-origin baseline evaluation. Your dataset must contain enough observations for the requested initial window and horizon.

## Python example

```python
import pandas as pd
from ts_intel.features import make_lag_features
from ts_intel.models.baselines import seasonal_naive
from ts_intel.metrics import forecast_metrics

series = pd.Series([10, 11, 12, 13, 14, 15, 16, 17])
prediction = seasonal_naive(series, horizon=2, season_length=4)
print(prediction)
```

## Architecture

```text
Raw observations
   |
   v
Validation / timestamp normalization
   |
   v
Temporal split + leakage-safe features
   |
   +----> Naive / seasonal naive / autoregression
   |
   +----> ML models / optional LSTM
   |
   v
Rolling-origin backtesting
   |
   +----> Point metrics (MAE, RMSE, sMAPE)
   +----> Residual diagnostics / anomaly scores
   +----> Change-point signals
   |
   v
Experiment record + reproducible report
```

## Important methodology

- Never randomly shuffle a time series for forecasting evaluation.
- Fit scalers, imputers, feature selectors, and models using training data only.
- Use rolling-origin or expanding-window backtesting.
- Compare against persistence and seasonal-naive baselines.
- Report forecast horizon, frequency, split dates, and metrics.
- Avoid claiming causal explanations from correlations or anomaly scores.
- For intermittent or seasonal series, choose metrics and baselines appropriate to the data.

## Repository structure

```text
src/ts_intel/
  data/          # validation and temporal splitting
  features/      # lag and rolling features
  models/        # naive and autoregressive baselines
  evaluation/    # backtesting
  anomaly/       # residual scoring and change points
  deep_learning/ # optional LSTM
  cli.py
tests/
configs/
docs/
```

## Limitations

The included methods are transparent baselines. Production forecasting needs domain-specific validation, calendar/event data, drift monitoring, uncertainty calibration, and operational decision thresholds. The LSTM module is optional and is not a pretrained forecasting system. No dataset or pretrained weights are included.

## Validation status

Archive integrity and required project files were checked. Unit tests, dependency installation, CLI execution, model training, and Docker build have **not** been run in this generated environment.
