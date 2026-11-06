import numpy as np

def forecast_metrics(actual, predicted):
    y, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("actual and predicted must have the same non-empty shape")
    error = y-p
    denominator = np.abs(y)+np.abs(p)
    smape = np.mean(np.divide(2*np.abs(error), denominator,
                              out=np.zeros_like(error), where=denominator != 0))*100
    return {"mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "smape": float(smape)}
