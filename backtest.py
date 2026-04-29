import json
import numpy as np
import pandas as pd
from fetch import fetch_btc
from model import predict_95
from calibrate import compute_scalar, CALIB_WINDOW

WARMUP = 72
GARCH_REFIT_EVERY = 24


def winkler_score(lo: float, hi: float, actual: float, alpha: float = 0.05) -> float:
    width = hi - lo
    if lo <= actual <= hi:
        return width
    elif actual < lo:
        return width + (2 / alpha) * (lo - actual)
    else:
        return width + (2 / alpha) * (actual - hi)


def run_backtest(output_path: str = "backtest_results.jsonl") -> dict:
    print("Fetching 750 bars of BTCUSDT 1h data...")
    df = fetch_btc(limit=750)
    df = df.iloc[-720:].reset_index(drop=True)
    total = len(df)
    print(f"Bars available: {total}  |  Warmup: {WARMUP}  |  Predictions: {total - WARMUP - 1}")

    predictions = []
    past_preds = []

    for i in range(WARMUP, total - 1):
        history = df.iloc[: i + 1]
        actual = float(df.iloc[i + 1]["close"])

        cache_key = (i // GARCH_REFIT_EVERY) * GARCH_REFIT_EVERY

        window = past_preds[-CALIB_WINDOW:] if len(past_preds) >= 20 else past_preds
        cal_scalar = compute_scalar(window)

        lo, hi = predict_95(history, cache_key=cache_key, cal_scalar=cal_scalar)

        hit = bool(lo <= actual <= hi)
        ws = winkler_score(lo, hi, actual)

        record = {
            "bar_index": i,
            "timestamp": df.iloc[i]["open_time"].isoformat(),
            "lo": round(lo, 2),
            "hi": round(hi, 2),
            "actual": round(actual, 2),
            "hit": hit,
            "width": round(hi - lo, 2),
            "winkler": round(ws, 2),
            "cal_scalar": round(cal_scalar, 4),
        }
        predictions.append(record)
        past_preds.append({"lo": lo, "hi": hi, "actual": actual})

        if i % 72 == 0 or i == total - 2:
            n = len(predictions)
            cov = sum(p["hit"] for p in predictions) / n
            avg_w = np.mean([p["width"] for p in predictions])
            print(f"  bar {i:>4}/{total}  n={n:>4}  coverage={cov:.3f}  avg_width=${avg_w:,.0f}  scalar={cal_scalar:.3f}")

    n = len(predictions)
    coverage = sum(p["hit"] for p in predictions) / n
    avg_width = np.mean([p["width"] for p in predictions])
    mean_winkler = np.mean([p["winkler"] for p in predictions])

    print(f"\n{'='*50}")
    print(f"BACKTEST COMPLETE")
    print(f"  N predictions  : {n}")
    print(f"  Coverage 95%   : {coverage:.4f}  (target: 0.95)")
    print(f"  Avg Width ($)  : {avg_width:,.2f}")
    print(f"  Mean Winkler   : {mean_winkler:,.2f}")
    print(f"{'='*50}")

    with open(output_path, "w") as f:
        for record in predictions:
            f.write(json.dumps(record) + "\n")
    print(f"Saved {n} predictions → {output_path}")

    metrics = {
        "n": n,
        "coverage_95": round(coverage, 4),
        "avg_width": round(avg_width, 2),
        "mean_winkler_95": round(mean_winkler, 2),
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics    → metrics.json")

    return metrics


if __name__ == "__main__":
    run_backtest()
