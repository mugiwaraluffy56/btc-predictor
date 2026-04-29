import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm, skew, kurtosis as sp_kurtosis

_garch_cache: dict[int, tuple[float, float]] = {}

_LOG2 = np.log(2)
_GK_C = 2 * _LOG2 - 1   # Garman-Klass constant ≈ 0.386


def garman_klass_sigma(df: pd.DataFrame, window: int = 24) -> float:
    """Per-bar volatility using OHLC — ~5× more efficient than close-only std."""
    n = min(window, len(df))
    tail = df.iloc[-n:]
    gk = 0.5 * np.log(tail["high"] / tail["low"]) ** 2 - _GK_C * np.log(tail["close"] / tail["open"]) ** 2
    return float(np.sqrt(gk.mean()))


def filter_jumps(log_returns: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Strip bars where |return| > threshold × σ before fitting."""
    sigma = log_returns.std()
    if sigma == 0:
        return log_returns
    cleaned = log_returns[np.abs(log_returns) <= threshold * sigma]
    return cleaned if len(cleaned) >= 30 else log_returns


def garch_vol(log_returns: np.ndarray, cache_key: int) -> tuple[float, float]:
    """
    GARCH(1,1) with Student-t errors. Returns (1-step sigma, fitted df_t).
    Result cached per cache_key — caller refreshes every 24 bars.
    """
    if cache_key in _garch_cache:
        return _garch_cache[cache_key]

    clean = filter_jumps(log_returns[-300:])
    try:
        am  = arch_model(clean * 100, vol="Garch", p=1, q=1, dist="t", rescale=False)
        res = am.fit(disp="off", show_warning=False, options={"maxiter": 200})
        var = res.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]
        sigma = float(np.sqrt(var)) / 100
        try:
            df_t = float(res.params["nu"])
        except KeyError:
            df_t = 5.0
        df_t = float(np.clip(df_t, 2.1, 30.0))
    except Exception:
        sigma = float(clean.std())
        df_t  = 5.0

    _garch_cache[cache_key] = (sigma, df_t)
    return sigma, df_t


def cf_quantile(log_returns: np.ndarray, p: float) -> float:
    """Cornish-Fisher expansion — adjusts Gaussian z-score for skewness and excess kurtosis."""
    z = norm.ppf(p)
    s = float(skew(log_returns))
    k = float(sp_kurtosis(log_returns, fisher=True))
    return z + (z**2 - 1)*s/6 + (z**3 - 3*z)*k/24 - (2*z**3 - 5*z)*s**2/36


def ensemble_vol(
    df_history: pd.DataFrame,
    log_ret: np.ndarray,
    cache_key: int,
) -> tuple[float, float]:
    """
    Blend Garman-Klass (40%) and GARCH (60%) volatility estimates.
    Accepts pre-computed log_ret to avoid duplicate diff(log(close)).
    Returns (blended sigma, fitted df_t).
    """
    gk_sigma    = garman_klass_sigma(df_history)
    garch_sigma, df_t = garch_vol(log_ret, cache_key)
    return 0.4 * gk_sigma + 0.6 * garch_sigma, df_t
