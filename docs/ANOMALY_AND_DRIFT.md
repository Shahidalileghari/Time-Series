# Anomaly detection and drift

Residual anomaly scoring assumes a reasonable reference forecast. A large residual is a review signal, not proof of fraud, fault, or root cause. Calibrate thresholds on validation data and account for seasonality and changing variance.

The rolling mean-shift method is an intentionally simple change-point baseline. For serious use, compare with CUSUM, PELT, Bayesian online change-point detection, and domain-specific alert policies. Monitor false-positive rate and alert fatigue.
