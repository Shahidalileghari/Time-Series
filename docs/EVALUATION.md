# Forecast evaluation protocol

Use chronological or rolling-origin splits. Do not shuffle. Report forecast horizon, sampling frequency, origin dates, baseline, and aggregated plus fold-level metrics.

- MAE: interpretable average absolute error.
- RMSE: penalizes large misses more heavily.
- sMAPE: scale-normalized percentage error; interpret cautiously near zero.
- MASE: recommended extension for comparing across scales using an in-sample naive denominator.

Compare persistence and seasonal naive before more complex models. For probabilistic forecasts, add pinball loss, interval coverage, and interval width. Inspect residual autocorrelation and error slices by season, regime, and event.
