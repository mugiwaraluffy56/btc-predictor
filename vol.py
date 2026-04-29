import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import skew, kurtosis as sp_kurtosis, norm

_garch_cache: dict = {}


def garman_klass_sigma(df: pd.DataFrame) -> float:
    log_hl = np.log(df["high"] / df["low"]) ** 2
    log_co = np.log(df["close"] / df["open"]) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    window = min(24, len(gk))
    return float(np.sqrt(gk.iloc[-window:].mean()))


def filter_jumps(log_returns: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    sigma = np.std(log_returns)
    if sigma == 0:
        return log_returns
    mask = np.abs(log_returns) <= threshold * sigma
    cleaned = log_returns[mask]
    return cleaned if len(cleaned) >= 30 else log_returns


def garch_vol(log_returns: np.ndarray, cache_key: int) -> tuple[float, float]:
    if cache_key in _garch_cache:
        return _garch_cache[cache_key]
    clean = filter_jumps(log_returns[-300:])
    scaled = clean * 100
    try:
        am = arch_model(scaled, vol="Garch", p=1, q=1, dist="t", rescale=False)
        res = am.fit(disp="off", show_warning=False, options={"maxiter": 200})
        fcast = res.forecast(horizon=1, reindex=False)
        sigma = float(np.sqrt(fcast.variance.iloc[-1, 0])) / 100
        df_t = float(res.params.get("nu", 5.0))
        df_t = max(2.1, min(df_t, 30.0))
    except Exception:
        sigma = float(np.std(clean))
        df_t = 5.0
    _garch_cache[cache_key] = (sigma, df_t)
    return sigma, df_t


def cf_quantile(log_returns: np.ndarray, p: float) -> float:
    z = norm.ppf(p)
    s = skew(log_returns)
    k = sp_kurtosis(log_returns, fisher=True)  # fisher=True gives excess kurtosis
    z_cf = (
        z
        + (z**2 - 1) * s / 6
        + (z**3 - 3 * z) * k / 24
        - (2 * z**3 - 5 * z) * s**2 / 36
    )
    return float(z_cf)


def ensemble_vol(df_history: pd.DataFrame, cache_key: int) -> tuple[float, float]:
    log_ret = np.diff(np.log(df_history["close"].values))
    gk_sigma = garman_klass_sigma(df_history)
    garch_sigma, df_t = garch_vol(log_ret, cache_key)
    blended = 0.4 * gk_sigma + 0.6 * garch_sigma
    return blended, df_t
