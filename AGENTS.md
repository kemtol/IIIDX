# AGENTS.md

Python 3.12 stock screener for Indonesian IDX equities. BSJP active, BPJS paused.

**ALWAYS read `program.md` first.** It defines the two-product architecture, invariants, session protocol, and current state. This file is the condensed reference.

## Two Products — Do Not Confuse

| | Training | Inference |
|---|---|---|
| **Output** | Model (`model_lightgbm_*.txt`) | Signal (top-3 picks daily) |
| **Code** | `edges/`, `generate_datamart.py`, `train_lightgbm.py` | `inferences/`, `fetch.py`, `run.py` |
| **Data** | L0 → L1 → L2 parquet | DuckDB (`inferences/bsjp/db/`) |
| **Schedule** | On-demand / after retrain | Daily cron 17:30 WIB |
| **Touch NEVER** | Don't touch `inferences/` | Don't touch `edges/` or L1/L2 |

## Venv

```bash
source ../.venv/bin/activate   # venv lives outside idx/
pip install -r requirements.txt
```

## Critical Invariants

**No look-ahead bias** — violating this invalidates everything:
- Broker features shifted by 1 day (`groupby(ticker,broker).shift(1)` at L1).
- Entry price is close@15:xx — must be known at decision time.
- Exit price is label only, never a feature.
- Objective must be explicit:
  - `overnight`: exit open@09 T+1.
  - `close10`: exit open@10 T+1.
- Walk-forward: strict chronological split, no randomization. OOT = last 100 trading days, used ONCE.
- AUC OOT in 0.55–0.62 is healthy. >0.65 = suspect leakage. >0.70 = almost certainly leakage.
- `min_data_in_leaf=100`, `lambda_l1/l2=1.0-1.5` is the sweet spot. `md=500` (v7 default) is over-regularized.
- **LGBMRanker failed** (v9d) — always use binary classifier.

## Data Layers

| Layer | Path | Grain |
|-------|------|-------|
| L0 Raw | `data/Level_0_Raw/` | per source |
| L1 Features | `data/Level_1_Features/broksum_datamart.parquet` | (date, broker, ticker) |
| L1 Modules | `data/Level_1_Features/modules/*_features.parquet` | (date, ticker) or (date,) |
| L2 Training | `data/Level_2_Datamart/` | (date, ticker) |
| Model | `model/BSJP/bsjp_vN/` | metrics.json, model .txt |
| **Inference** | `inferences/bsjp/db/inference.duckdb` | (date, ticker) |

**Inference reads DuckDB, NOT L1/L2 parquet.** Never trigger L1/L2 rebuild from inference code.

## Commands

### Training (research/retrain)

```bash
bash pipeline/run/run_fetch_broksum.sh      # ~30-60 min, resume-capable
bash pipeline/run/run_feature_l1.sh
cd edges/bsjp_overnight_sl2/scripts
python generate_datamart.py --exit-hour 10
python train_lightgbm.py \
  --output-dir ../../model/BSJP/bsjp_vN \
  --feature-modules-dir ../../data/Level_1_Features/modules \
  --feature-prune-top-n 0 \
  --tp-pct 0.01 --sl-pct -0.02 \
  --oot-valid-days 100
```

### Inference (daily production)

```bash
# Cron (after all L0 fetchers complete):
bash pipeline/run/run_inference_bsjp.sh

# Manual:
cd inferences/bsjp/golang && go build -o bsjp ./cmd/bsjp/
./bsjp fetch --date $(date +%Y-%m-%d) --force
./bsjp predict --variant v19d_close10_preclose14_orb_md100_l21.5 --log
./bsjp predict --variant v15 --log
```

### Go binary

```bash
cd inferences/bsjp/golang && go build -o bsjp ./cmd/bsjp/

# One-time bootstrap (from existing parquet L0):
./bsjp bootstrap

# Daily fetch (rebuilds features_store for latest date):
./bsjp fetch

# Preflight check (L0 + DuckDB + model readiness):
./bsjp check --verbose --telegram   # sends to Telegram + Discord

# Fetch 1h bars from Yahoo Finance (optional, cron handles L0):
./bsjp download --limit 100

# Historical deployed research variant; do not use its backtest as v23 evidence:
./bsjp predict --variant v19d_close10_preclose14_orb_md100_l21.5 --log

# Predict v15 (no ARA policy):
./bsjp predict --variant v15 --log
```

### Calibration (Go vs Python)

```bash
cd inferences/bsjp/golang
go build -o bsjp ./cmd/bsjp/
GOTOOLCHAIN=local go test ./internal/features/ -v -run TestCalibrate 2>&1 | tee /tmp/cal.log
```

### Tests

```bash
pytest pipeline/storage/tests/test_continuity.py -v   # merge gate, 4 tests
```

## Feature Modules (Preferred Fast Path)

`--feature-modules-dir data/Level_1_Features/modules` loads precomputed feature parquets via LEFT JOIN instead of monolithic rebuild. Add features by dropping `*_features.parquet` into `modules/`; do not delete feature columns for pruning, exclude them at train time.

Current v23 research uses the expanded module set and 315 selected model features. Check `model/BSJP/v23b_t1audit2_clean/feature_importance.csv` and `metrics.json` before assuming old 252-feature behavior.

## Current BSJP Research Handoff (2026-05-09)

- **Current clean candidate:** `model/BSJP/v23b_t1audit2_clean/`.
- **Objective:** BSJP `close10` — entry close 15:xx T, exit open 10:xx T+1.
- **Status:** research-only. Do not treat this as production or inference-ready.
- **Locked OOT artifact:** `valid_predictions.parquet`, 100 trading days, 2025-11-17 → 2026-04-23.
- **Locked OOT metrics:** AUC 0.5346, cum net +207.9%, MaxDD -24.2%, best iteration 5, overfit gap 0.0616.
- **No-lookahead audit:** `_LOG/v23b_t1audit2_clean_no_lookahead_audit_20260509.json`; hard failures all false.
- **Current policy candidate:** k=2 / max weight 25% / cost cap 3% / adaptive q=.85. Locked OOT +255.0%, MaxDD -17.9%, active days 96.
- **Conservative policy candidate:** k=2 / max weight 20% / q=.90. Locked OOT +184.7%, MaxDD -13.5%.
- **Latest local closed-date extension:** provisional scoring to 2026-05-06. 2026-05-09 is Saturday; 2026-05-08 entry is not closed yet.
- **Rp10m provisional calendar extension:** 7D +11.31%, 30D +22.48%, 90D +171.51%, all ending 2026-05-06.
- **Caveat:** post-2026-04-23 extension is not locked OOT; broker aggregate/CVD modules currently end at 2026-04-23, so rows after that have missing broker-family values.

Plain read: v23 is not a pure ARA hunter. Gain is mostly pre14 intraday (~67%) and macro prev-close (~27%); ARA-history is small (~2.6%). Remaining concern is robustness/overfit, not a confirmed leakage failure in the current audit scope.

## Model Version Warnings

- **v15** — production inference variant (AUC 0.602, overnight, clean).
- **v19d/v20** — historical ARA-continuation baseline/policy, but headline returns are inflated by confirmed same-day broker leakage and ARA fillability assumptions. Do not use as credibility benchmark.
- **v23b_t1audit2_clean** — current clean research candidate after broker/CVD/yf_daily/ARA-history audits. Not production; needs rolling-retrain validation.
- **v18 close10 rebuild** — historical reference only. Depends on close15/EOD features unavailable before 14:59 decision.
- **Close10 v10/grid models** — **DATA LEAKAGE**: `close_ret_last1h` leaked as feature. INVALID; do not use.
- **OLD `training_datamart_bsjp_overnight.parquet`** — misleading filename; labels are actually `bsjp_close10_sl2` (exit open@10). Do not call close10 artifacts `overnight`.
- **`training_datamart_bsjp_overnight_fixed.parquet`** — hybrid forensic artifact, not clean canonical training data.
- **BPJS** — PAUSED. 1h window too narrow to clear break-even.

See `model/BSJP/LATEST.md` for full iteration history.

## L0 → DuckDB Migration (PRD 0003)

- **Phase 0: done.** Storage abstraction library complete (`pipeline/storage/`): writers, readers, validators (DuckDB `%` quoting, timestamp normalization), 3-tier backup, migration CLI, continuity gate.
- **Phase 1 quick win: done.** `master.duckdb` populated (773 emiten + 92 broker rows), validator parity 0 diffs.
- **Active invariant:** default canary flags = stage 0 (parquet only). All current pipelines untouched until env vars (`L0_*_DUCKDB_WRITE=true`) promote individual sources.
- **Storage auto-registration:** a new schema dropped into `pipeline/storage/schemas/<source>.py` + registered in `schemas/__init__.py` is automatically picked up by writers, readers, validators, migration, and continuity gate. Verified end-to-end on `master_broker`.
- **Continuity gate** (`test_continuity.py`): must pass before any fetch script refactor. Currently 4 parametrized tests across 2 schemas.
- **Next:** refactor `pipeline/fetch/fetch_emiten.py` and `fetch_master_broker_idx.py` to call `write_l0()` instead of `df.to_parquet()`.

## Go Parity Gaps

- `yf_daily.go` — fixed (PERFECT match after `.JK` suffix strip). 58/58 calibrated.
- `overnight.go` — fixed (rewrote from 1h to daily parquet). 24/27 PERFECT, 3 cols p10 epsilon.
- `broker_agg.go` — base flow features calibrated (18/18 PERFECT vs fresh Python). Timeflow/broker-type/context features (~96 cols) from bootstrap (stale), not yet regenerated in Go.
- Stockbit/XL — implemented (4 cols, 98/98 PERFECT).
- Preclose14 volume — fixed (min_periods in rollingMAShifted). Daily vol/turnover MA 6/6 PERFECT. 2 sparse tickers only.
- `preclose14.go` — VWAP, ORB, ARA-state features already Go-native via preclose14 module (115 cols).
- `fetch_lightweight.py` — archived. Not needed for Go inference path.
- `run_inference_bsjp.sh` — fixed (Go binary now).

## Known Issues

- v23 robustness is not proven yet: best iteration is 5 and rolling-retrain validation is still pending.
- Post-2026-04-23 calendar extension has missing broker-family values because broker aggregate/CVD module coverage ends at 2026-04-23.
- Cross-sectional features (`sq_`, `xc_`, `yp_`, `pd_`) have NOT been audited for lookahead safety.
- `ipot_ohlcv_1h.parquet` has zero live consumers — PRD 0003 deferred; fetcher still runs but removed from inference cron.
- Training scripts pre-27-Apr were lost; always version-lock with git.
- Broker timeflow/context features (~96 cols) not regenerated in Go — rely on bootstrap values.
- L0 data is not version-locked between training and inference — small distribution shift possible.

## Conventions

- **All logs go to `_LOG/`** — not `edges/.../` or `inferences/.../`.
- `float32` dtype for all feature columns — must be consistent training↔inference.
- **No hardcoded paths** — use repo-root-relative paths. Shell scripts use `$IDX_DIR`.
- After training, update `model/BSJP/LATEST.md`. Never delete old models — move to `model/BSJP/_ARCH/`.
- Session notes: write to `_MEMORY/YYYYMMDDHHMMSS.md`. Do NOT leave TODO comments in code.
- Broker fetcher: `run_fetch_broksum.sh` has resume capability (`_LOG/broksum_resume_state.json`). Run after 17:35 WIB.
