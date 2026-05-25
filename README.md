# CMA Dashboard — ARCHIVED

> **Status: Archived.** The monthly TimesFM + dashboard pipeline has been
> retired. The automated schedule is disabled. The full pipeline can still
> be run manually via `workflow_dispatch` if needed.

## What remains active

**`credit_signal.py`** — a standalone 60-line script that computes the
GEP-evolved credit signal z-score. No PyTorch, no TimesFM, no model
downloads. Runs in ~3 seconds.

```bash
pip install numpy pandas requests yfinance
python credit_signal.py
```

Output:
```
Credit Signal — 2026-04-30
  HYG 6M vol:  0.0101
  T10Y2Y:      +0.50% pts
  Raw signal:  0.7172
  z-score:     -0.33
  Bucket:      NEUTRAL
```

## Why archived

The full dashboard recommended a Barbell Permanent Portfolio allocation
(SPY/TLT/SHY/GLD/TIP with HYG tactical overlay) that was subsequently
replaced by a fiscal-dominance-oriented portfolio. Specifically:

- **Credit signal (t-stat 1.11):** Not statistically significant. The
  tactical excess was +0.19% CAGR — noise.
- **TimesFM CMA forecasts:** After recentering to historical means, the
  forecasts converge to the historical mean by construction. A spreadsheet
  with trailing returns provides equivalent information.
- **PP allocation recommendation:** Rejected in favor of a portfolio with
  no TLT (fiscal-dominance concern), no HYG (CLOs instead), STIP instead
  of TIP, and factor-tilted ex-US equity instead of SPY.
- **Maintenance cost:** 9 dependencies including PyTorch (~800MB), a
  git-installed research model, and 6 external data sources — each a
  potential silent-failure point.

## Research value preserved

- `src/dashboard.py` — full pipeline code with 32 bug fixes applied
- `docs/history.csv` — signal log archive
- `docs/cma_latest.csv` — last CMA forecast snapshot
- `docs/index.html` + `docs/figures/` — last rendered dashboard

The GEP signal derivation (pooled symbolic regression over HYG/LQD/MUB
2008–2024, converging on `vol_6m + sqrt(T10Y2Y)`) and the barbell PP
design are documented in the parent research repo.

## Original layout

```
cma-dashboard/
├── credit_signal.py              # ACTIVE — lightweight signal monitor
├── src/
│   └── dashboard.py              # ARCHIVED — full pipeline
├── docs/                         # last dashboard snapshot
│   ├── index.html
│   ├── figures/
│   ├── cma_latest.csv
│   └── history.csv
├── requirements.txt              # for full pipeline only
└── .github/workflows/
    └── dashboard.yml             # schedule disabled; manual trigger only
```
