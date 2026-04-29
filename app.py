import json
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fetch import fetch_btc
from model import predict_95
from calibrate import compute_scalar

PREDICTIONS_FILE = "predictions.jsonl"
BACKTEST_METRICS_FILE = "metrics.json"
GARCH_REFIT_EVERY = 24

st.set_page_config(page_title="BTC Predictor — AlphaI", layout="wide")

# ── Backtest metrics ───────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_metrics() -> dict:
    try:
        with open(BACKTEST_METRICS_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"coverage_95": "N/A", "avg_width": "N/A", "mean_winkler_95": "N/A", "n": "N/A"}


# ── Part C: persistence ────────────────────────────────────────────────────
def load_past_predictions() -> list[dict]:
    try:
        with open(PREDICTIONS_FILE) as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def save_prediction(record: dict):
    with open(PREDICTIONS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def rewrite_predictions(records: list[dict]):
    with open(PREDICTIONS_FILE, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def fill_actuals(past: list[dict], df: pd.DataFrame) -> list[dict]:
    ts_to_close = {
        row["open_time"].strftime("%Y-%m-%dT%H:00:00+00:00"): row["close"]
        for _, row in df.iterrows()
    }
    changed = False
    for p in past:
        if "actual" not in p and p.get("next_bar_ts") in ts_to_close:
            p["actual"] = round(ts_to_close[p["next_bar_ts"]], 2)
            p["hit"] = bool(p["lo"] <= p["actual"] <= p["hi"])
            changed = True
    if changed:
        rewrite_predictions(past)
    return past


# ── Main ───────────────────────────────────────────────────────────────────
st.title("BTC/USDT 1h Prediction — AlphaI × Polaris")

metrics = load_metrics()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Coverage (Backtest)", f"{metrics['coverage_95']}")
c2.metric("Avg Range Width ($)", f"${metrics['avg_width']:,}" if isinstance(metrics['avg_width'], (int, float)) else metrics['avg_width'])
c3.metric("Mean Winkler Score", f"{metrics['mean_winkler_95']}")
c4.metric("Backtest Predictions", f"{metrics['n']}")

st.divider()

with st.spinner("Fetching live BTC data from Binance..."):
    df = fetch_btc(limit=500)

current_price = float(df["close"].iloc[-1])
current_ts    = df["open_time"].iloc[-1]
next_bar_ts   = (current_ts + pd.Timedelta("1h")).strftime("%Y-%m-%dT%H:00:00+00:00")

# Rolling calibration scalar from past live predictions with actuals
past = load_past_predictions()
past = fill_actuals(past, df)
past_with_actual = [p for p in past if "actual" in p]
cal_scalar = compute_scalar(past_with_actual[-48:])

cache_key = (len(df) // GARCH_REFIT_EVERY) * GARCH_REFIT_EVERY
lo, hi = predict_95(df, cache_key=cache_key, cal_scalar=cal_scalar)

# Deduplicate: only save if no prediction already exists for this bar
already_saved = any(p.get("timestamp", "")[:13] == current_ts.isoformat()[:13] for p in past)
if not already_saved:
    record = {
        "timestamp":   current_ts.isoformat(),
        "next_bar_ts": next_bar_ts,
        "current":     round(current_price, 2),
        "lo":          round(lo, 2),
        "hi":          round(hi, 2),
    }
    save_prediction(record)
    past.append(record)

# ── Live prediction ────────────────────────────────────────────────────────
st.subheader(f"Current BTC Price: ${current_price:,.2f}")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Predicted Low (next 1h)", f"${lo:,.2f}")
col_b.metric("Predicted High (next 1h)", f"${hi:,.2f}")
col_c.metric("Range Width", f"${hi - lo:,.2f}")
st.caption(f"Bar closed at: {current_ts.strftime('%Y-%m-%d %H:%M UTC')}  |  Calibration scalar: {cal_scalar:.3f}")

st.divider()

# ── Chart: last 50 bars + predicted ribbon ─────────────────────────────────
chart_df   = df.iloc[-50:].reset_index(drop=True)
next_time  = current_ts + pd.Timedelta("1h")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=chart_df["open_time"],
    y=chart_df["close"],
    mode="lines",
    name="BTC Close",
    line=dict(color="#F7931A", width=2),
))

# Shade the predicted range for the next bar
fig.add_trace(go.Scatter(
    x=[current_ts, next_time, next_time, current_ts, current_ts],
    y=[hi, hi, lo, lo, hi],
    fill="toself",
    fillcolor="rgba(99, 179, 237, 0.2)",
    line=dict(color="rgba(99, 179, 237, 0.8)", width=1, dash="dot"),
    name="95% Predicted Range",
))

# Mark the midpoint
mid = (lo + hi) / 2
fig.add_trace(go.Scatter(
    x=[current_ts, next_time],
    y=[current_price, mid],
    mode="lines",
    line=dict(color="rgba(99,179,237,0.5)", width=1, dash="dash"),
    name="Midpoint projection",
    showlegend=False,
))

fig.add_annotation(x=next_time, y=hi, text=f"  Hi ${hi:,.0f}", showarrow=False,
                   xanchor="left", font=dict(color="#63b3ed", size=11))
fig.add_annotation(x=next_time, y=lo, text=f"  Lo ${lo:,.0f}", showarrow=False,
                   xanchor="left", font=dict(color="#63b3ed", size=11))

fig.update_layout(
    title="Last 50 Bars + Next-Hour Predicted Range",
    xaxis_title="Time (UTC)",
    yaxis_title="Price (USD)",
    template="plotly_dark",
    height=480,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(r=100),
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Part C: prediction history ─────────────────────────────────────────────
if past:
    st.subheader(f"Live Prediction History ({len(past)} entries)")

    live_hits   = [p for p in past if "actual" in p and p.get("hit")]
    live_misses = [p for p in past if "actual" in p and not p.get("hit")]
    live_cov    = len(live_hits) / len([p for p in past if "actual" in p]) if any("actual" in p for p in past) else None

    if live_cov is not None:
        lc1, lc2, lc3 = st.columns(3)
        lc1.metric("Live Coverage", f"{live_cov:.3f}")
        lc2.metric("Live Hits", len(live_hits))
        lc3.metric("Live Misses", len(live_misses))

    rows = []
    for p in reversed(past[-100:]):
        hit_str = "✓ Hit" if p.get("hit") else ("✗ Miss" if "actual" in p else "⏳ Pending")
        rows.append({
            "Timestamp (UTC)": p["timestamp"][:16].replace("T", " "),
            "BTC at Pred":     f"${p['current']:,.2f}",
            "Lo":              f"${p['lo']:,.2f}",
            "Hi":              f"${p['hi']:,.2f}",
            "Actual":          f"${p['actual']:,.2f}" if "actual" in p else "—",
            "Result":          hit_str,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.caption(f"Auto-refreshes every 60 seconds · Data: Binance BTCUSDT 1h · Model: GARCH(1,1) + Garman-Klass + Cornish-Fisher")
time.sleep(1)
st.rerun()
