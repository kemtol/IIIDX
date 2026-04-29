# AGENTS.md

Python 3.12 stock screener for Indonesian IDX equities. Two strategies (BSJP active, BPJS paused).

**ALWAYS read `program.md` first.** It defines the two-product architecture, invariants, and session protocol. This file is the condensed reference.

## Two Products — Do Not Confuse

| | Training | Inference |
|---|---|---|
| **Output** | Model (`model_lightgbm_*.txt`) | Signal (top-3 picks daily) |
| **Code** | `edges/`, `generate_datamart.py`, `train_lightgbm.py` | `inferences/`, `fetch.py`, `run.py` |
| **Data** | L0 → L1 → L2 parquet | DuckDB (`inferences/bsjp/db/`) |
| **Schedule** | On-demand / after retrain | Daily cron 17:30 WIB |
| **Deploy** | Dev machine | VPS 4GB RAM |
| **Touch NEVER** | Don't touch `inferences/` | Don't touch `edges/` or L1/L2 |

## Venv

```bash
source ../.venv/bin/activate   # venv lives outside idx/
pip install -r requirements.txt
```

## Critical Invariants

**No look-ahead bias** — the most important rule:
- Broker features shifted by 1 day (T-1 features predict T outcome).
- Entry price is close@15:xx (BSJP) — must be known at decision time.
- Exit price (open T+1 @09:05) is label only, never a feature.
- Walk-forward validation: strict chronological split, no randomization.

## Data Layers

| Layer | Path | Grain |
|-------|------|-------|
| Level 0 (Raw) | `data/Level_0_Raw/` | per source |
| Level 1 (Features) | `data/Level_1_Features/broksum_datamart.parquet` | (date, broker, ticker) |
| Level 1 (Modules) | `data/Level_1_Features/modules/*_features.parquet` | (date, ticker) or (date,) |
| Level 2 (Training) | `data/Level_2_Datamart/training_datamart_bsjp_overnight.parquet` | (date, ticker) |
| Model artifacts | `model/BSJP/bsjp_vN/` | metrics.json, model .txt |
| **Inference (DuckDB)** | `inferences/bsjp/db/inference.duckdb` | (date, ticker) |

**Production inference uses DuckDB, NOT L1/L2 parquet rewrite.**

## Training (batch, for research/retrain)

```bash
bash pipeline/run/run_fetch_broksum.sh
bash pipeline/run/run_feature_l1.sh
cd edges/bsjp_overnight_sl2/scripts
python generate_datamart.py --strategy-mode bsjp
python train_lightgbm.py \
  --output-dir ../../model/BSJP/bsjp_vN \
  --feature-modules-dir ../../data/Level_1_Features/modules \
  --feature-prune-top-n 0 \
  --tp-pct 0.01 --sl-pct -0.02 \
  --oot-valid-days 100
```

## Inference (daily production)

```bash
# One-time bootstrap:
python inferences/bsjp/python/bootstrap_feature_store.py --replace

# Daily cron:
python inferences/bsjp/python/fetch.py
python inferences/bsjp/python/run.py --variant v15 --log-picks
```

**Alternative: Go binary (no Python runtime)**

```bash
inferences/bsjp/golang/bsjp bootstrap
inferences/bsjp/golang/bsjp fetch
inferences/bsjp/golang/bsjp predict --variant v15 --log
```

**DO NOT** trigger L1/L2 rebuild from `fetch.py` on a cron. The default fast path reads L0 and upserts to DuckDB without touching parquet.

## Model Versions

- **BSJP v15** — current active (AUC 0.602, MaxDD -15.3%, IHSG MA). Clean datamart, Monte Carlo validated.
- **BSJP v7** — reference baseline (AUC 0.602, MaxDD -30%).
- **Close10 models** (v10, grid search) — **DATA LEAKAGE**: leaked `exit_price`/`overnight_return`/`close_ret_last1h` as features. INVALID, re-run needed with clean datamart.
- **BPJS** — PAUSED. 1h window too narrow to clear break-even.

## Feature Modules (Preferred Fast Path)

Training with `--feature-modules-dir data/Level_1_Features/modules` loads precomputed feature parquets via LEFT JOIN (5-10s) instead of monolithic 3-min rebuild. Byte-identical feature set. Add new features by dropping a `*_features.parquet` into modules dir.

## Current State & Priorities

See `program.md` §4 and `model/BSJP/LATEST.md` for full details. Key priorities:
1. Paper trade v15
2. Re-run grid search with clean datamart
3. Fix Go inference parity (overnight + yf_daily calibration)
4. Fix Python `fetch_lightweight.py` merge bug

## Known Issues

- Close10 models and grid search have data leakage — re-train needed with clean datamart.
- LGBMRanker failed (v9d) — use binary classifier.
- `min_data_in_leaf=500` (v7 default) is over-regularized; `md=100, λ=1.0-1.5` is sweet spot.
- VWAP and HMM features not needed for BSJP v15 inference path.
- `fetch_lightweight.py` has duplicate column merge bug (670 lines, blocked).

## Directory Structure for New Strategies

```
edges/<name>/
  edge.md              # spec, iterations, lessons learned
  scripts/
    generate_datamart.py   # L1 → L2 aggregation + labeling
    train_lightgbm.py      # walk-forward training + simulation
  analysis/            # notebooks for post-hoc evaluation
```

Level 0 and Level 1 are shared across edges.

## Broker Activity Fetcher

`run_fetch_broksum.sh` has resume capability (`_LOG/broksum_resume_state.json`). Runtime 30-60 minutes. Run after market close (~17:35 WIB).
