import numpy as np

def residual_anomaly_scores(actual, predicted, scale_floor=1e-8):
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if a.shape != p.shape: raise ValueError("actual and predicted shapes must match")
    residual = a-p
    center = np.median(residual)
    mad = np.median(np.abs(residual-center))
    robust_scale = max(1.4826*mad, scale_floor)
    return np.abs(residual-center)/robust_scale

def flag_anomalies(scores, threshold=3.5):
    if threshold <= 0: raise ValueError("threshold must be positive")
    return np.asarray(scores, dtype=float) >= threshold
