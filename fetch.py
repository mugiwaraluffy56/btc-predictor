import requests
import pandas as pd
import numpy as np

BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"


def fetch_btc(limit: int = 1000, symbol: str = "BTCUSDT", interval: str = "1h") -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(BINANCE_URL, params=params, timeout=10)
    r.raise_for_status()
    raw = r.json()
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[["open_time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    diffs = df["open_time"].diff().dropna()
    assert (diffs > pd.Timedelta("90min")).sum() == 0, "Gap in BTC data detected"
    return df
