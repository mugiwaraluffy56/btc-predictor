import numpy as np

CALIB_WINDOW = 48
TARGET_COVERAGE = 0.95


def compute_scalar(predictions_window: list[dict]) -> float:
    if len(predictions_window) < 20:
        return 1.0

    actuals = np.array([p["actual"] for p in predictions_window])
    los     = np.array([p["lo"]     for p in predictions_window])
    his     = np.array([p["hi"]     for p in predictions_window])
    mids    = (los + his) / 2
    halfs   = (his - los) / 2

    lo_s, hi_s = 0.5, 3.0
    for _ in range(40):
        s = (lo_s + hi_s) / 2
        hits = ((actuals >= mids - halfs * s) & (actuals <= mids + halfs * s)).mean()
        if hits < TARGET_COVERAGE:
            lo_s = s
        else:
            hi_s = s
    return float((lo_s + hi_s) / 2)
