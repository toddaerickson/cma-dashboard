#!/usr/bin/env python3
"""Lightweight credit signal monitor.

Computes the GEP-evolved credit signal z-score from two inputs:
  - HYG 6-month realized volatility (via VWEHX proxy)
  - 10y-2y term spread (FRED T10Y2Y)

Signal: vol_6m(HYG) + sqrt(max(T10Y2Y, 0))
Z-score: expanding-window normalization (min 36 months)
Buckets: ON (z >= +0.5), NEUTRAL, OFF (z <= -0.5)

Run monthly after market close:
  python3 credit_signal.py
"""
from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# --- HYG proxy returns ---
px = yf.download("VWEHX", start="1998-01-01", auto_adjust=True, progress=False)
if "Close" in px.columns:
    rets = px["Close"].resample("ME").last().pct_change()
else:
    rets = px.iloc[:, 0].resample("ME").last().pct_change()
vol6 = rets.rolling(6).std()

# --- FRED T10Y2Y ---
resp = requests.get(
    "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y",
    timeout=30,
)
resp.raise_for_status()
t10y2y = pd.read_csv(
    StringIO(resp.text), parse_dates=["observation_date"],
).rename(columns={"observation_date": "date"}).set_index("date")["T10Y2Y"]
t10y2y = pd.to_numeric(t10y2y, errors="coerce").resample("ME").last()

# --- Signal ---
# Units intentionally mixed: vol6 is decimal (~0.01), T10Y2Y is
# percentage points (~0.5). This is the exact GEP-evolved form.
sig = vol6 + np.sqrt(np.clip(t10y2y, 0, None))
sig = sig.dropna()
mu = sig.expanding(min_periods=36).mean()
sd = sig.expanding(min_periods=36).std()
z = (sig - mu) / sd

z_now = float(z.iloc[-1])
asof = z.index[-1].date()

if not np.isfinite(z_now):
    print(f"ERROR: z-score is {z_now} for {asof}. Check inputs.")
    raise SystemExit(1)

if z_now >= 0.5:
    bucket = "ON"
elif z_now <= -0.5:
    bucket = "OFF"
else:
    bucket = "NEUTRAL"

vol_now = float(vol6.iloc[-1])
term_now = float(t10y2y.iloc[-1])
raw_now = float(sig.iloc[-1])

print(f"Credit Signal — {asof}")
print(f"  HYG 6M vol:  {vol_now:.4f}")
print(f"  T10Y2Y:      {term_now:+.2f}% pts")
print(f"  Raw signal:  {raw_now:.4f}")
print(f"  z-score:     {z_now:+.2f}")
print(f"  Bucket:      {bucket}")
