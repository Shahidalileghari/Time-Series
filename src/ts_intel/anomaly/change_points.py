import numpy as np

def rolling_mean_shift(values, window=12, threshold=2.5):
    """Simple change signal: adjacent rolling means separated by pooled local stds."""
    x = np.asarray(values, dtype=float)
    if window < 2 or len(x) < 2*window:
        raise ValueError("Need at least 2*window values and window >= 2")
    signals = []
    for split in range(window, len(x)-window+1):
        left, right = x[split-window:split], x[split:split+window]
        pooled = np.sqrt((left.var()+right.var())/2)
        score = abs(right.mean()-left.mean()) / max(pooled, 1e-8)
        if score >= threshold: signals.append({"index": split, "score": float(score)})
    return signals
