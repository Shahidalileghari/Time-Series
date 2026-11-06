# Architecture

The package separates data validation, feature generation, model baselines, evaluation, anomaly analysis, and CLI orchestration. Optional deep-learning models are isolated so core workflows do not require GPU libraries.

## Leakage boundary

A feature at time t may only use observations available at or before t-1 when predicting t. The lag feature builder shifts before rolling. Any learned preprocessing must be fitted separately within each training fold.

## Extension points

Add model adapters implementing a consistent `fit(history)` / `forecast(horizon)` contract; add probabilistic outputs as quantile arrays or prediction intervals; add model-specific tuning only inside each training fold.
