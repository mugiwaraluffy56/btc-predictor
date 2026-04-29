import numpy as np
import pandas as pd
from vol import ensemble_vol, cf_quantile, filter_jumps

N_SIMS = 50_000
RNG = np.random.default_rng(42)


def predict_95(
    df_history: pd.DataFrame,
    cache_key: int = -1,
    cal_scalar: float = 1.0,
) -> tuple[float, float]:
    assert len(df_history) >= 72, f"Need at least 72 bars, got {len(df_history)}"

    log_ret = np.diff(np.log(df_history["close"].values))
    clean_ret = filter_jumps(log_ret)

    sigma, df_t = ensemble_vol(df_history, cache_key)
    mu = float(np.mean(log_ret[-24:]))
    S0 = float(df_history["close"].iloc[-1])

    eps = RNG.standard_t(df=df_t, size=N_SIMS)
    log_S1 = (mu - 0.5 * sigma**2) + sigma * eps
    S1 = S0 * np.exp(log_S1)

    raw_lo, raw_hi = np.percentile(S1, [2.5, 97.5])

    cf_lo_z = cf_quantile(clean_ret, 0.025)
    cf_hi_z = cf_quantile(clean_ret, 0.975)
    cf_lo = S0 * np.exp((mu - 0.5 * sigma**2) + sigma * cf_lo_z)
    cf_hi = S0 * np.exp((mu - 0.5 * sigma**2) + sigma * cf_hi_z)

    lo = min(raw_lo, cf_lo)
    hi = max(raw_hi, cf_hi)

    mid = (lo + hi) / 2
    half = (hi - lo) / 2 * cal_scalar
    return float(mid - half), float(mid + half)
