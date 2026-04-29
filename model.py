import numpy as np
import pandas as pd
from vol import ensemble_vol, cf_quantile, filter_jumps

N_SIMS = 50_000
_RNG   = np.random.default_rng(42)


def predict_95(
    df_history: pd.DataFrame,
    cache_key: int = -1,
    cal_scalar: float = 1.0,
) -> tuple[float, float]:
    """
    Predict the 95% interval for the next bar close.

    No future data must be present in df_history — callers are responsible
    for passing only bars up to and including the current bar.
    """
    if len(df_history) < 72:
        raise ValueError(f"predict_95 needs ≥72 bars, got {len(df_history)}")

    log_ret   = np.diff(np.log(df_history["close"].values))
    clean_ret = filter_jumps(log_ret)
    S0        = float(df_history["close"].iloc[-1])
    mu        = float(log_ret[-24:].mean())

    sigma, df_t = ensemble_vol(df_history, log_ret, cache_key)
    drift       = mu - 0.5 * sigma ** 2

    # Monte Carlo simulation
    eps  = _RNG.standard_t(df=df_t, size=N_SIMS)
    S1   = S0 * np.exp(drift + sigma * eps)
    sim_lo, sim_hi = float(np.percentile(S1, 2.5)), float(np.percentile(S1, 97.5))

    # Cornish-Fisher analytic bounds (uses higher moments of actual returns)
    cf_lo = S0 * np.exp(drift + sigma * cf_quantile(clean_ret, 0.025))
    cf_hi = S0 * np.exp(drift + sigma * cf_quantile(clean_ret, 0.975))

    # Take the more conservative of simulation vs analytic
    lo, hi = min(sim_lo, cf_lo), max(sim_hi, cf_hi)

    # Apply calibration scalar symmetrically around midpoint
    mid  = (lo + hi) / 2
    half = (hi - lo) / 2 * cal_scalar
    return mid - half, mid + half
