"""Paired bootstrap with exact batch quota and simultaneous contrast intervals."""
import numpy as np


def bootstrap_gain(scores, D, tie, counts, fraction=.2, chunk=128):
    order = np.lexsort((tie, -scores))
    d = D[order].astype(np.int32)
    n = len(D)
    if counts.ndim != 2 or counts.shape[1] != n or not np.all(counts.sum(axis=1) == n):
        raise ValueError("Each resample must have exactly n requests")
    k = int(fraction*n)
    output = np.zeros(len(counts))
    if not k:
        return output
    for begin in range(0, len(counts), chunk):
        c = counts[begin:begin+chunk, order].astype(np.int32)
        total = np.cumsum(c, axis=1, dtype=np.int32)
        gain = np.cumsum(c*d, axis=1, dtype=np.int32)
        row = np.arange(len(c))
        boundary = (total >= k).argmax(axis=1)
        previous = np.maximum(boundary-1, 0)
        nc = np.where(boundary > 0, total[row, previous], 0)
        ng = np.where(boundary > 0, gain[row, previous], 0)
        output[begin:begin+len(c)] = (ng+(k-nc)*d[boundary])/n
    return output


def simultaneous_intervals(point, draws):
    se = draws.std(axis=0, ddof=1)
    safe = np.maximum(se, 1e-12)
    max_t = np.abs((draws-point)/safe).max(axis=1)
    critical = np.quantile(max_t, .95, method="higher")
    p = (1+(max_t[:, None] >= np.abs(point/safe)).sum(axis=0))/(len(draws)+1)
    return {"se": se, "low": point-critical*se, "high": point+critical*se,
            "adjusted_p": p, "critical": float(critical)}
