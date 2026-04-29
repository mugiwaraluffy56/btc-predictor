import json
import os
import tempfile
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from calibrate import compute_scalar
from fetch import fetch_btc
from model import predict_95

PREDICTIONS_FILE    = "predictions.jsonl"
BACKTEST_METRICS    = "metrics.json"
GARCH_REFIT_EVERY   = 24
REFRESH_INTERVAL_S  = 60

st.set_page_config(page_title="BTC Predictor — AlphaI", layout="wide")


# ── Helpers ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400)
def load_metrics() -> dict:
    try:
        with open(BACKTEST_METRICS) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"coverage_95": "N/A", "avg_width": "N/A", "mean_winkler_95": "N/A", "n": "N/A"}


def _fmt_metric(val: object, prefix: str = "") -> str:
    return f"{prefix}{val:,}" if isinstance(val, (int, float)) else str(val)


def load_predictions() -> list[dict]:
    try:
        with open(PREDICTIONS_FILE) as f:
            return [json.loads(ln) for ln in f if ln.strip()]
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def append_prediction(record: dict) -> None:
    with open(PREDICTIONS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def persist_predictions(records: list[dict]) -> None:
    """Atomic write — swap temp file to avoid partial reads."""
    dir_ = os.path.dirname(os.path.abspath(PREDICTIONS_FILE)) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, PREDICTIONS_FILE)
    except Exception:
        os.unlink(tmp)
        raise


def fill_actuals(past: list[dict], df: pd.DataFrame) -> tuple[list[dict], bool]:
    """Backfill actual closes for past predictions once their bar has closed."""
    ts_map = dict(zip(
        df["open_time"].dt.strftime("%Y-%m-%dT%H:00:00+00:00"),
        df["close"].round(2),
    ))
    changed = False
    for p in past:
        if "actual" not in p:
            actual = ts_map.get(p.get("next_bar_ts", ""))
            if actual is not None:
                p["actual"] = actual
                p["hit"]    = bool(p["lo"] <= actual <= p["hi"])
                changed     = True
    return past, changed


def build_chart(chart_df: pd.DataFrame, current_ts: pd.Timestamp, lo: float, hi: float) -> go.Figure:
    next_ts = current_ts + pd.Timedelta("1h")
    mid     = (lo + hi) / 2
    S0      = float(chart_df["close"].iloc[-1])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_df["open_time"], y=chart_df["close"],
        mode="lines", name="BTC Close",
        line=dict(color="#F7931A", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[current_ts, next_ts, next_ts, current_ts, current_ts],
        y=[hi, hi, lo, lo, hi],
        fill="toself", fillcolor="rgba(99,179,237,0.18)",
        line=dict(color="rgba(99,179,237,0.75)", width=1, dash="dot"),
        name="95% Predicted Range",
    ))
    fig.add_trace(go.Scatter(
        x=[current_ts, next_ts], y=[S0, mid],
        mode="lines", showlegend=False,
        line=dict(color="rgba(99,179,237,0.4)", width=1, dash="dash"),
    ))
    for y, label in [(hi, f"Hi ${hi:,.0f}"), (lo, f"Lo ${lo:,.0f}")]:
        fig.add_annotation(x=next_ts, y=y, text=f"  {label}", showarrow=False,
                           xanchor="left", font=dict(color="#63b3ed", size=11))
    fig.update_layout(
        title="Last 50 Bars + Next-Hour Predicted Range",
        xaxis_title="Time (UTC)", yaxis_title="Price (USD)",
        template="plotly_dark", height=480, margin=dict(r=110),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── Page ───────────────────────────────────────────────────────────────────

st.title("BTC/USDT 1h Prediction — AlphaI × Polaris")

# Backtest headline metrics
m = load_metrics()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Coverage (Backtest)",   _fmt_metric(m["coverage_95"]))
c2.metric("Avg Range Width ($)",   _fmt_metric(m["avg_width"], "$"))
c3.metric("Mean Winkler Score",    _fmt_metric(m["mean_winkler_95"]))
c4.metric("Backtest Predictions",  _fmt_metric(m["n"]))
st.divider()

# Live data
with st.spinner("Fetching live BTC data…"):
    try:
        df = fetch_btc(limit=500)
    except Exception as exc:
        st.error(f"Binance fetch failed: {exc}")
        st.stop()

current_price = float(df["close"].iloc[-1])
current_ts    = df["open_time"].iloc[-1]
next_bar_ts   = (current_ts + pd.Timedelta("1h")).strftime("%Y-%m-%dT%H:00:00+00:00")

# Part C: load history, backfill actuals
past         = load_predictions()
past, dirty  = fill_actuals(past, df)
if dirty:
    persist_predictions(past)

# Rolling calibration from resolved predictions
resolved   = [p for p in past if "actual" in p]
cal_scalar = compute_scalar(resolved[-48:])

# Prediction
cache_key  = (len(df) // GARCH_REFIT_EVERY) * GARCH_REFIT_EVERY
lo, hi     = predict_95(df, cache_key=cache_key, cal_scalar=cal_scalar)

# Deduplicate by hour prefix
saved_hours = {p["timestamp"][:13] for p in past}
if current_ts.isoformat()[:13] not in saved_hours:
    rec = {
        "timestamp":   current_ts.isoformat(),
        "next_bar_ts": next_bar_ts,
        "current":     round(current_price, 2),
        "lo":          round(lo, 2),
        "hi":          round(hi, 2),
    }
    append_prediction(rec)
    past.append(rec)

# Live prediction display
st.subheader(f"Current BTC Price: ${current_price:,.2f}")
ca, cb, cc = st.columns(3)
ca.metric("Predicted Low  (next 1h)", f"${lo:,.2f}")
cb.metric("Predicted High (next 1h)", f"${hi:,.2f}")
cc.metric("Range Width",              f"${hi - lo:,.2f}")
st.caption(
    f"Bar: {current_ts.strftime('%Y-%m-%d %H:%M UTC')}  ·  "
    f"Calibration scalar: {cal_scalar:.3f}  ·  "
    f"Next bar: {next_bar_ts[:16].replace('T',' ')} UTC"
)
st.divider()

# Chart
st.plotly_chart(build_chart(df.iloc[-50:].reset_index(drop=True), current_ts, lo, hi),
                use_container_width=True)
st.divider()

# Part C: history table
if past:
    settled = [p for p in past if "actual" in p]
    st.subheader(f"Live Prediction History ({len(past)} entries)")

    if settled:
        hits     = sum(p["hit"] for p in settled)
        live_cov = hits / len(settled)
        h1, h2, h3 = st.columns(3)
        h1.metric("Live Coverage",  f"{live_cov:.3f}")
        h2.metric("Resolved Hits",  hits)
        h3.metric("Resolved Misses", len(settled) - hits)

    rows = [
        {
            "Timestamp (UTC)": p["timestamp"][:16].replace("T", " "),
            "BTC at Pred":     f"${p['current']:,.2f}",
            "Lo":              f"${p['lo']:,.2f}",
            "Hi":              f"${p['hi']:,.2f}",
            "Actual":          f"${p['actual']:,.2f}" if "actual" in p else "—",
            "Result":          ("✓ Hit" if p["hit"] else "✗ Miss") if "actual" in p else "⏳ Pending",
        }
        for p in reversed(past[-100:])
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.caption(
    "Auto-refreshes every 60 s  ·  "
    "Data: Binance BTCUSDT 1h  ·  "
    "Model: GARCH(1,1) + Garman-Klass + Cornish-Fisher"
)

time.sleep(REFRESH_INTERVAL_S)
st.rerun()
