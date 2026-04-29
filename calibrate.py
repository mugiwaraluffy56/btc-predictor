import numpy as np

CALIB_WINDOW    = 48
TARGET_COVERAGE = 0.95
_SCALAR_BOUNDS  = (0.5, 3.0)
_BISECT_ITERS   = 40


def compute_scalar(predictions_window: list[dict]) -> float:
    """
    Binary-search for the half-width multiplier that achieves TARGET_COVERAGE
    on the supplied window of past predictions (each must have lo, hi, actual).
    Returns 1.0 until ≥20 resolved predictions are available.
    """
    if len(predictions_window) < 20:
        return 1.0

    actuals = np.fromiter((p["actual"] for p in predictions_window), float)
    mids    = np.fromiter(((p["lo"] + p["hi"]) / 2 for p in predictions_window), float)
    halfs   = np.fromiter(((p["hi"] - p["lo"]) / 2 for p in predictions_window), float)

    lo_s, hi_s = _SCALAR_BOUNDS
    for _ in range(_BISECT_ITERS):
        s    = (lo_s + hi_s) / 2
        hits = np.mean((actuals >= mids - halfs * s) & (actuals <= mids + halfs * s))
        if hits < TARGET_COVERAGE:
            lo_s = s
        else:
            hi_s = s

    return float((lo_s + hi_s) / 2)
