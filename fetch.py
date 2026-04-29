import time
import requests
import pandas as pd

BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"
_OHLCV_COLS  = ["open", "high", "low", "close", "volume"]
_RAW_COLS    = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore",
]


def fetch_btc(
    limit: int = 1000,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    retries: int = 3,
) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    last_exc: Exception = RuntimeError("unreachable")

    for attempt in range(retries):
        try:
            r = requests.get(BINANCE_URL, params=params, timeout=10)
            r.raise_for_status()
            break
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"Binance fetch failed after {retries} attempts: {last_exc}") from last_exc

    df = pd.DataFrame(r.json(), columns=_RAW_COLS)
    df[_OHLCV_COLS] = df[_OHLCV_COLS].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[["open_time"] + _OHLCV_COLS].reset_index(drop=True)

    gaps = df["open_time"].diff().dropna()
    if (gaps > pd.Timedelta("90min")).any():
        raise ValueError("Gap > 90 min detected in BTC data — fetch a larger window")

    return df
