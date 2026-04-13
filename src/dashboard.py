"""Monthly dashboard generator for the PP + GEP credit-overlay strategy.

Run on the first business day of each month after market close:
  python3 gep_eval/dashboard.py

Produces:
  dashboard/index.html          — single-page self-contained dashboard
  dashboard/figures/            — fresh PNGs
  dashboard/history.csv         — appended signal log (as-of date, z, bucket, weights)

Pipeline:
  1. Refresh month-end prices for the 8 base assets + proxies + BTC/etc.
  2. Pull FRED T10Y2Y (month-end).
  3. Compute current credit signal:  vol_6m(HYG) + sqrt(max(t10y2y, 0))
     -> expanding z-score; bucket {on | neutral | off}.
  4. Derive recommended weights for this month:
       PP base:            SPY 25 / TLT 25 / GLD 25 / TIP 25
       Tactical overlay:   bucket determines HYG share stolen from TIP.
  5. Re-run TimesFM CMA forecasts for all assets.
  6. Recenter medians on long-run historical mean.
  7. Regenerate figures + write HTML dashboard.
"""
from __future__ import annotations
import json
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "docs"
FIG  = DASH / "figures"
for d in (DASH, FIG):
    d.mkdir(exist_ok=True)

PROXIES = {"SPY": "SPY", "TLT": "VUSTX", "GLD": "GC=F",
           "TIP": "VIPSX", "HYG": "VWEHX", "IEF": "VFITX",
           "SHY": "VFISX"}
CMA_UNIVERSE = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ",
    "TLT", "IEF", "LQD", "HYG", "MUB", "TIP",
    "GLD", "SLV", "USO", "USL", "DBC", "DBA",
    "UUP", "FXF", "FXE", "FXY",
    "BTC-USD",
    "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLB", "XLU", "KBE",
]

# -------------------------------------------------------------------
# 1. Signal + allocation
# -------------------------------------------------------------------
def compute_signal_and_weights():
    px = yf.download(list(PROXIES.values()), start="1998-01-01",
                     auto_adjust=True, progress=False)["Close"]
    m = px.resample("ME").last()
    data = pd.DataFrame({k: m[v] for k, v in PROXIES.items()})
    rets = data.pct_change().dropna(how="all")

    t10y2y = pd.read_csv(
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y",
        parse_dates=["observation_date"]
    ).rename(columns={"observation_date": "date"}).set_index("date")["T10Y2Y"]
    t10y2y = pd.to_numeric(t10y2y, errors="coerce").resample("ME").last()

    vol6 = rets["HYG"].rolling(6).std()
    sig_raw = vol6 + np.sqrt(np.clip(t10y2y, 0, None))
    sig_raw = sig_raw.dropna()
    mu = sig_raw.expanding(min_periods=36).mean()
    sd = sig_raw.expanding(min_periods=36).std()
    z = (sig_raw - mu) / sd

    asof = z.index[-1]
    z_now = float(z.iloc[-1])
    # Base: barbell PP (25 SPY / 10 TLT / 15 SHY / 25 GLD / 25 TIP).
    # Tactical overlay steals from TIP for HYG based on credit signal.
    base = {"SPY": 0.25, "TLT": 0.10, "SHY": 0.15, "GLD": 0.25, "TIP": 0.25, "HYG": 0.00}
    if z_now >= 0.5:
        bucket = "ON"
        w = {**base, "TIP": 0.10, "HYG": 0.15}
    elif z_now <= -0.5:
        bucket = "OFF"
        w = {**base, "TIP": 0.25, "HYG": 0.00}
    else:
        bucket = "NEUTRAL"
        w = {**base, "TIP": 0.175, "HYG": 0.075}

    return {
        "asof": asof, "z": z_now, "bucket": bucket, "weights": w,
        "sig_raw_now": float(sig_raw.iloc[-1]),
        "t10y2y_now": float(t10y2y.loc[:asof].iloc[-1]),
        "hyg_vol6m_now": float(vol6.loc[:asof].iloc[-1]),
        "z_history": z,
        "rets_history": rets,
    }


# -------------------------------------------------------------------
# 2. CMA: run TimesFM + recenter
# -------------------------------------------------------------------
def run_cma():
    import torch, timesfm
    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(timesfm.ForecastConfig(
        max_context=1024, max_horizon=64, normalize_inputs=True,
        use_continuous_quantile_head=True, force_flip_invariance=True,
        infer_is_positive=False, fix_quantile_crossing=True,
    ))
    px = yf.download(CMA_UNIVERSE, start="1998-01-01",
                     auto_adjust=True, progress=False)["Close"]
    monthly = px.resample("ME").last()
    rows = []
    for t in CMA_UNIVERSE:
        if t not in monthly.columns: continue
        s = np.log(monthly[t].dropna())
        if len(s) < 36: continue
        ctx = s.values.astype(np.float32)
        if len(ctx) > 1024: ctx = ctx[-1024:]
        point, q = model.forecast(horizon=6, inputs=[ctx])
        base = float(s.iloc[-1])
        qs = q[0][-1]  # horizon-6 quantiles: [mean, q10..q90]
        d = {"ticker": t,
             "mean": float(np.exp(qs[0] - base) - 1),
             "q10":  float(np.exp(qs[1] - base) - 1),
             "q50":  float(np.exp(qs[5] - base) - 1),
             "q90":  float(np.exp(qs[9] - base) - 1),
             "n":    int(len(s))}
        # historical 6M mean for recentering
        r6 = np.expm1(np.log(monthly[t].dropna()).diff().rolling(6).sum()).dropna()
        hmean = float(r6.iloc[:-6].mean()) if len(r6) > 6 else np.nan
        shift = hmean - d["q50"]
        for k in ("q10", "q50", "q90", "mean"):
            d[k + "_adj"] = d[k] + shift
        d["hist_mean_6m"] = hmean
        d["shift"] = shift
        rows.append(d)
    return pd.DataFrame(rows).set_index("ticker")


# -------------------------------------------------------------------
# 3. Figures
# -------------------------------------------------------------------
def make_figures(sig, cma):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams.update({"figure.dpi": 110, "font.size": 10})

    # Signal z over time with bucket bands
    fig, ax = plt.subplots(figsize=(9.5, 3.5))
    z = sig["z_history"].dropna()
    ax.plot(z.index, z.values, lw=1.2, color="#2060c0")
    ax.axhspan(0.5, 4, color="#c03030", alpha=0.09, label="ON (z ≥ +0.5)")
    ax.axhspan(-0.5, 0.5, color="#888", alpha=0.09, label="NEUTRAL")
    ax.axhspan(-4, -0.5, color="#208030", alpha=0.09, label="OFF (z ≤ -0.5)")
    ax.axhline(sig["z"], color="#c03030", lw=1.2, ls="--")
    ax.set_ylim(z.min() - 0.2, z.max() + 0.2)
    ax.set_ylabel("Credit-signal z-score")
    ax.set_title(f"Credit signal history (as of {sig['asof'].date()}) — current bucket: {sig['bucket']}")
    ax.legend(loc="lower left", ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    plt.tight_layout()
    plt.savefig(FIG / "signal_history.png", bbox_inches="tight"); plt.close()

    # CMA quantiles
    c = cma.sort_values("q50_adj")
    fig, ax = plt.subplots(figsize=(9, 8))
    y = np.arange(len(c))
    ax.barh(y, (c["q90_adj"] - c["q10_adj"]) * 100, left=c["q10_adj"] * 100,
            color="#c0d4ed", edgecolor="#7a9ec9", lw=0.5, height=0.6)
    ax.scatter(c["q50_adj"] * 100, y, color="#c03030", s=28, zorder=5)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels(c.index, fontsize=9)
    ax.set_xlabel("6-month forward total return (%)")
    ax.set_title("CMA distributions (80% PI, historical-mean centered)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "cma.png", bbox_inches="tight"); plt.close()

    # Allocation pie
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    w = sig["weights"]
    labels = [f"{k}\n{v*100:.1f}%" for k, v in w.items() if v > 0]
    values = [v for v in w.values() if v > 0]
    colors_ = ["#2060c0", "#c03030", "#c09030", "#208030", "#9040a0"]
    ax.pie(values, labels=labels, colors=colors_[:len(values)],
           startangle=90, wedgeprops={"edgecolor": "white", "lw": 1.5})
    ax.set_title(f"Current recommended allocation ({sig['bucket']})")
    plt.tight_layout()
    plt.savefig(FIG / "allocation.png", bbox_inches="tight"); plt.close()


# -------------------------------------------------------------------
# 4. HTML render
# -------------------------------------------------------------------
HTML_TMPL = """<!doctype html>
<html><head><meta charset="utf-8">
<title>PP + GEP Credit Overlay — {asof}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,Helvetica,Arial,sans-serif;
     max-width:1100px;margin:20px auto;padding:0 20px;color:#222;line-height:1.45}}
h1{{border-bottom:2px solid #c03030;padding-bottom:6px}}
h2{{margin-top:32px;color:#444}}
.banner{{padding:14px 18px;border-radius:6px;margin:16px 0;font-size:15px}}
.bucket-on{{background:#fae3e3;border-left:5px solid #c03030}}
.bucket-neutral{{background:#eee;border-left:5px solid #888}}
.bucket-off{{background:#e3f0e3;border-left:5px solid #208030}}
table{{border-collapse:collapse;margin:10px 0}}
td,th{{padding:6px 12px;border-bottom:1px solid #ddd;text-align:right}}
th{{background:#f4f4f4;text-align:left}}
.num{{font-variant-numeric:tabular-nums}}
img{{max-width:100%;height:auto;margin:10px 0}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.footnote{{color:#666;font-size:12px;margin-top:40px;border-top:1px solid #ddd;padding-top:10px}}
</style></head>
<body>
<h1>Barbell Permanent Portfolio + GEP Credit Overlay</h1>
<div style="color:#666;font-size:12px">Base: 25 SPY / 10 TLT / 15 SHY / 25 GLD / 25 TIP &nbsp;·&nbsp;
HYG tactical tilt from TIP</div>
<div class="num">As of <b>{asof}</b> &nbsp;·&nbsp; generated {generated}</div>

<div class="banner bucket-{bucket_css}">
<b>Current signal: {bucket}</b> &nbsp;·&nbsp;
credit-signal z = <b>{z:+.2f}</b>
(raw signal {sig_raw:.4f}; HYG 6M vol {vol:.4f}; 10y-2y {term:+.2f}%)
<br>
Rule: z ≥ +0.5 → ON (HYG 15% from TIP);
−0.5 &lt; z &lt; +0.5 → NEUTRAL (HYG 7.5% / TIP 17.5%);
z ≤ −0.5 → OFF (HYG 0% / TIP 25%).
SPY 25 / TLT 10 / SHY 15 / GLD 25 held constant.
</div>

<div class="two-col">
<div>
<h2>Recommended allocation</h2>
<img src="figures/allocation.png">
<table class="num">
<tr><th>Asset</th><th>Weight</th></tr>
{weights_rows}
</table>
</div>
<div>
<h2>Signal history</h2>
<img src="figures/signal_history.png">
</div>
</div>

<h2>CMA distributions (6-month forward total return)</h2>
<img src="figures/cma.png">
<p style="color:#666;font-size:12px">80% prediction intervals from TimesFM 2.5 with current-regime volatility;
medians recentered to historical 6-month mean per asset. Use for risk budgeting / scenario analysis,
not for tactical allocation.</p>

<h2>Core portfolio performance reference (2008-01 → 2026-04)</h2>
<table class="num">
<tr><th>Strategy</th><th>CAGR</th><th>Vol</th><th>Sharpe</th><th>Max DD</th><th>2008</th><th>2022</th></tr>
<tr><td><b>Barbell PP (base)</b></td><td>+7.2%</td><td>7.6%</td><td>0.95</td><td>-16.4%</td><td>-6.0%</td><td>-11.3%</td></tr>
<tr><td>Classic PP</td><td>+7.7%</td><td>8.5%</td><td>0.90</td><td>-18.9%</td><td>-4.1%</td><td>-15.2%</td></tr>
<tr><td>Intl-duration PP</td><td>+7.2%</td><td>8.5%</td><td>0.85</td><td>-18.6%</td><td>-5.8%</td><td>-14.1%</td></tr>
<tr><td>60/40 (SPY/IEF)</td><td>+8.3%</td><td>9.6%</td><td>0.87</td><td>-27.7%</td><td>-18.7%</td><td>-14.8%</td></tr>
<tr><td>SPY B&amp;H</td><td>+10.8%</td><td>15.6%</td><td>0.69</td><td>-46.3%</td><td>-36.8%</td><td>-18.2%</td></tr>
</table>
<p style="color:#666;font-size:12px">Barbell replaces 15pp of TLT with SHY.
Buys 4pp of 2022 protection and 15% lower vol for ~0.5pp of CAGR — a favorable
trade in a fiscal-dominance / steepening-curve regime.</p>
<p style="color:#666;font-size:12px">Backtested on Vanguard mutual-fund proxies where ETFs
didn't yet exist (VUSTX, VWEHX, VIPSX, etc.). 5 bps/side transaction cost.</p>

<div class="footnote">
Strategy rationale &amp; negative-result log: see <code>gep_eval/README.md</code>.<br>
Credit signal derived from pooled GEP on HYG/LQD/MUB (2008-2024), expression
<code>vol_6m + sqrt(t10y2y)</code> — in-sample p&lt;0.001, live t-stat 1.11.
Treat tactical tilt as a small, low-confidence overlay on the PP base.<br>
Dashboard auto-regenerates monthly via cron (see <code>cron.md</code>).
</div>

</body></html>
"""


def render_html(sig, cma):
    wrows = "\n".join(
        f"<tr><td>{k}</td><td>{v*100:.1f}%</td></tr>"
        for k, v in sig["weights"].items() if v > 0
    )
    html = HTML_TMPL.format(
        asof=sig["asof"].date().isoformat(),
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        bucket=sig["bucket"],
        bucket_css=sig["bucket"].lower(),
        z=sig["z"], sig_raw=sig["sig_raw_now"],
        vol=sig["hyg_vol6m_now"], term=sig["t10y2y_now"],
        weights_rows=wrows,
    )
    (DASH / "index.html").write_text(html)


# -------------------------------------------------------------------
# 5. Append to history log
# -------------------------------------------------------------------
def log_history(sig):
    row = {
        "asof": sig["asof"].date().isoformat(),
        "generated": datetime.now().isoformat(timespec="seconds"),
        "z": sig["z"], "bucket": sig["bucket"],
        "sig_raw": sig["sig_raw_now"],
        "t10y2y": sig["t10y2y_now"], "hyg_vol6m": sig["hyg_vol6m_now"],
        **{f"w_{k}": v for k, v in sig["weights"].items()},
    }
    path = DASH / "history.csv"
    df = pd.DataFrame([row])
    if path.exists():
        prev = pd.read_csv(path)
        # skip if same asof already logged
        if row["asof"] not in prev["asof"].astype(str).values:
            df = pd.concat([prev, df], ignore_index=True)
        else:
            df = prev
    df.to_csv(path, index=False)


# -------------------------------------------------------------------
def main():
    print("1. signal + weights …")
    sig = compute_signal_and_weights()
    print(f"   asof={sig['asof'].date()}  z={sig['z']:+.2f}  bucket={sig['bucket']}")

    print("2. TimesFM CMA forecasts …")
    cma = run_cma()
    cma.to_csv(DASH / "cma_latest.csv")

    print("3. figures …")
    make_figures(sig, cma)

    print("4. HTML dashboard …")
    render_html(sig, cma)

    print("5. history log …")
    log_history(sig)

    print(f"\nDashboard written to: {DASH/'index.html'}")
    print(f"Open with: file://{(DASH/'index.html').resolve()}")


if __name__ == "__main__":
    main()
