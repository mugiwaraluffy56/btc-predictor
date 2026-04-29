import numpy as np

CALIB_WINDOW = 48
TARGET_COVERAGE = 0.95
SCALAR_EMA_ALPHA = 0.15   # smoothing — lower = slower adaptation, less jumpiness

_ema_scalar: float = 1.0  # module-level EMA state


def compute_scalar(predictions_window: list[dict]) -> float:
    global _ema_scalar
    if len(predictions_window) < 20:
        return 1.0

    actuals = np.array([p["actual"] for p in predictions_window])
    los     = np.array([p["lo"]     for p in predictions_window])
    his     = np.array([p["hi"]     for p in predictions_window])
    mids    = (los + his) / 2
    halfs   = (his - los) / 2

    # Binary search for raw scalar
    lo_s, hi_s = 0.5, 3.0
    for _ in range(40):
        s = (lo_s + hi_s) / 2
        hits = ((actuals >= mids - halfs * s) & (actuals <= mids + halfs * s)).mean()
        if hits < TARGET_COVERAGE:
            lo_s = s
        else:
            hi_s = s
    raw_scalar = (lo_s + hi_s) / 2

    # EMA smooth to prevent overcorrection jumps
    _ema_scalar = SCALAR_EMA_ALPHA * raw_scalar + (1 - SCALAR_EMA_ALPHA) * _ema_scalar
    return float(_ema_scalar)
