# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

**IMPORTANT — Start every session by reading `program.md`.** It is the single source of truth for product boundaries, invariants, current state, and agent protocol.

## Project Summary

MMMACHINE/idx — fully-automated stock screener for Indonesian IDX equities. Two products:

- **Training** → produces LightGBM models (`edges/`, `model/`)
- **Inference** → produces daily top-3 picks (`inferences/`)

One active strategy: **BSJP** (Beli Sore Jual Pagi) — entry close 15:xx, exit open 09:xx T+1.

## Environment

```bash
source ../.venv/bin/activate   # venv lives outside idx/
pip install -r requirements.txt
```

## Before Working

Read these files first:
1. `program.md` — product boundaries, invariants, current state
2. `model/BSJP/LATEST.md` — model history, decisions, forensics
3. `_MEMORY/` (latest file) — last session context

## During Work

- Know which product you're working on: Training or Inference. NEVER mix them.
- If TRAINING: work in `edges/`, `model/`, `data/Level_2_Datamart/`
- If INFERENCE: work in `inferences/`, `pipeline/run/run_inference_bsjp.sh`
- Never trigger L1/L2 rebuild from inference code
- All logs go to `_LOG/`

## After Work

- Write session notes to `_MEMORY/YYYYMMDDHHMMSS.md`
- Update `model/BSJP/LATEST.md` if model iterations changed
- Update `program.md` if product boundaries changed

## Quick Commands

```bash
# Fetch broker data (after market close ~17:35)
bash pipeline/run/run_fetch_broksum.sh

# Build features + training datamart
bash pipeline/run/run_feature_l1.sh
cd edges/bsjp_overnight_sl2/scripts
python generate_datamart.py --strategy-mode bsjp

# Train (with feature modules — fast path)
python train_lightgbm.py \
  --output-dir ../../model/BSJP/bsjp_vN \
  --feature-modules-dir ../../data/Level_1_Features/modules \
  --feature-prune-top-n 0 \
  --tp-pct 0.01 --sl-pct -0.02 \
  --oot-valid-days 100

# Inference (daily production)
python inferences/bsjp/python/fetch.py
python inferences/bsjp/python/run.py --variant v15 --log-picks

# Check logs
tail -f _LOG/*.log
```

## Tech Stack

| Layer | Library |
|-------|---------|
| Data fetching | `aiohttp`, `playwright`, `websockets` |
| Data processing | `pandas>=2.0`, `pyarrow>=14.0` |
| Storage | Parquet (data lake), DuckDB (inference store) |
| ML | `lightgbm`, `scikit-learn` |
| Inference (future) | Go binary with embedded DuckDB |
