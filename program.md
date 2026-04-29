# program.md — MMMACHINE/idx Autonomous Development Protocol

*This is the single source of truth. Every agent session starts here. The human edits this to improve the development process over time — just like Karpathy's autoresearch `program.md`, but adapted for a two-product financial ML system instead of a single train.py.*

---

## 0. Agent Protocol (Read This First)

### When starting a new session, ALWAYS read in this order:

1. **`program.md`** (this file) — product boundaries, current state, invariants
2. **`model/BSJP/LATEST.md`** — narrative context: model iterations, decisions, forensics
3. **`_MEMORY/`** (latest file) — what happened last session, what's next
4. **Target product's PRD** — `_DOC/_PRD/` or `inferences/PLANNING.md` or `edges/bsjp_overnight_sl2/edge.md`

### When ending a session, ALWAYS update:

1. **`_MEMORY/YYYYMMDDHHMMSS.md`** — date, scope, decisions, changes, next priority
2. **`model/BSJP/LATEST.md`** — if model iterations changed or new findings
3. **`program.md`** — if product boundaries, invariants, or current state changed

### Critical rule: KNOW WHICH PRODUCT YOU'RE WORKING ON

Before touching any file, ask: "Is this Training or Inference?" If you can't answer, stop and re-read this document.

---

## 1. Product Architecture

This project builds **two separate products**. They share data layers but have different code, different invariants, different deployment targets, and different output artifacts. NEVER conflate them.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SHARED DATA LAYERS                            │
│                                                                      │
│  Level 0 (Raw)                Level 1 (Features)                    │
│  data/Level_0_Raw/            data/Level_1_Features/                │
│  ├── broksum_bybroker.parquet ├── broksum_datamart.parquet          │
│  ├── yfinance_1h.parquet      ├── modules/*_features.parquet        │
│  ├── yfinance_daily.parquet   └── vwap_features.parquet             │
│  ├── global_indices.parquet                                          │
│  ├── master_broker.parquet                                           │
│  └── master_emiten.parquet                                           │
└──────────────┬──────────────────────────────────┬───────────────────┘
               │                                  │
               ▼                                  ▼
┌──────────────────────────────┐  ┌──────────────────────────────────┐
│   PRODUCT 1: TRAINING        │  │   PRODUCT 2: INFERENCE           │
│   Output: MODEL              │  │   Output: SIGNAL SERVICE         │
│                              │  │                                  │
│  Entry points:               │  │  Entry points:                   │
│  generate_datamart.py        │  │  fetch.py (or Go binary)         │
│  train_lightgbm.py           │  │  run.py (or Go binary)           │
│                              │  │                                  │
│  Artifacts:                  │  │  Artifacts:                      │
│  model/BSJP/bsjp_vN/         │  │  inferences/bsjp/db/*.duckdb     │
│  ├── metrics.json            │  │  inferences/bsjp/variants/       │
│  ├── model_lightgbm_*.txt    │  │  Picks log + daily signals       │
│  ├── feature_importance.csv  │  │                                  │
│  ├── valid_predictions.parquet│  │  Invariants:                     │
│  └── monte_carlo/            │  │  - Byte-identical to training    │
│                              │  │  - < 5 min runtime               │
│  Invariants:                 │  │  - NO L1/L2 rebuild on cron      │
│  - No lookahead bias         │  │  - Read from L0 or DuckDB only   │
│  - Walk-forward validation   │  │  - Single binary deployable      │
│  - Strict chronological split│  │                                  │
│  - OOT = 100 trading days    │  │  Target: VPS 4GB RAM             │
│                              │  │  Schedule: 17:30 WIB Mon-Fri     │
│  Schedule: on-demand /       │  │                                  │
│  research-driven. After      │  │  End user: signal consumer       │
│  market close >= 17:35 WIB.  │  │  (Telegram / API)                │
└──────────────────────────────┘  └──────────────────────────────────┘
```

### Product 1: Training (produces MODEL)

**What it is:** Research pipeline that takes raw market data, engineers features, trains LightGBM binary classifiers, and validates through walk-forward simulation + Monte Carlo.

**The "model" is the deliverable** — a single `model_lightgbm_*.txt` file consumed by Product 2.

**When to work on this:**
- Iterating on a new model version (vN)
- Adding features / feature families
- Grid search / hyperparameter optimization
- Debugging data leakage
- Running Monte Carlo validation

**Do NOT touch:** `inferences/`, DuckDB, cron scripts, Go binary.

### Product 2: Inference (produces SIGNAL SERVICE)

**What it is:** Production pipeline that reads precomputed (or compute-on-the-fly) feature vectors for today's date, scores them with the production model, and outputs top-3 picks.

**The "signal" is the deliverable** — a ranked list of 3 tickers with entry prices, sent to end users daily.

**When to work on this:**
- Improving fetch performance (Go binary, lightweight compute)
- Fixing DuckDB freshness / bootstrap
- Deploying to VPS
- Adding monitoring / alerting
- Changing variant config (k, weights, thresholds)

**Do NOT touch:** `edges/`, `generate_datamart.py`, `train_lightgbm.py`, L1/L2 parquet files.

### Training → Inference handoff (CRITICAL)

When a new model version is promoted to production:
1. Training produces `model/BSJP/bsjp_vN/model_lightgbm_*.txt`
2. Inference reads this file — **no code changes needed in inference**
3. If feature list changes: re-bootstrap DuckDB from L2 parquet
4. Update `model/BSJP/LATEST.md` and `program.md` §4

---

## 2. Critical Invariants (Applies to Both Products)

### No Lookahead Bias — VIOLATING THIS INVALIDATES EVERYTHING

| Rule | Enforcement |
|------|-------------|
| Broker features shifted by 1 day | `groupby(ticker, broker).shift(1)` at L1 level |
| Entry price = close@15:xx (BSJP) — must be known at decision time | Label construction verifies timestamp |
| Exit price = open@09:xx T+1 — is LABEL ONLY, never a feature | `OUTCOME_COLS` blocked in training |
| Walk-forward: strict chronological split, no randomization | `TimeSeriesSplit` or manual fold boundaries |
| OOT window = most recent 100 trading days | Never train on OOT, never use OOT for param selection |

### Data Leakage Prevention Checklist

Before trusting any model result:
- [ ] All broker columns use `shift(1)` → verified by checking max(date) vs target date
- [ ] `exit_price`, `overnight_return`, `close_ret_last1h` (for close10) are NOT in feature set
- [ ] Feature importance: #1 feature gain should NOT be >10x #2 (indicates leakage)
- [ ] AUC OOT in 0.55-0.62 range. >0.65 = SUSPECT, >0.70 = almost certainly leakage
- [ ] Overfit gap (train AUC − OOT AUC) < 0.05 for model with 200+ features
- [ ] OOT window genuinely untouched — not used for any param selection

### Walk-Forward Discipline

```
Fold 1: Train [day 0..N-400], Valid [N-399..N-300], OOT [N-299..N-200]
Fold 2: Train [day 0..N-300], Valid [N-299..N-200], OOT [N-199..N-100]
Fold 3: Train [day 0..N-200], Valid [N-199..N-100], OOT [N-99..N]
```

- Valid set = used for early stopping only, NOT for param selection
- OOT set = used ONCE for final evaluation. Never use for anything else.
- If you grid search 43 param combos and pick best by OOT → hyperparameter overfitting

### Data Layer Grain (Do Not Confuse)

| Layer | Path | Grain | Rows | Created By |
|-------|------|-------|------|------------|
| L0 Raw | `data/Level_0_Raw/` | per source | — | `pipeline/fetch/` |
| L1 Features | `data/Level_1_Features/` | (date, broker, ticker) | ~122 cols | `pipeline/feature/` |
| L1 Modules | `data/Level_1_Features/modules/` | (date, ticker) or (date,) | 252 cols | `generate_datamart.py` |
| L2 Training | `data/Level_2_Datamart/` | (date, ticker) | ~260 cols | `generate_datamart.py` |
| L2 Fixed | `data/Level_2_Datamart/*_fixed.parquet` | (date, ticker) | ~260 cols | Manual patching |
| DuckDB Store | `inferences/bsjp/db/` | (date, ticker) | ~253 cols | `fetch.py` / `bootstrap_*.py` |

**L1 and L2 ARE TRAINING ARTIFACTS. Inference reads DuckDB, not L1/L2.**

---

## 3. Directory Structure (Standardized)

```
idx/
├── program.md              ← THIS FILE — single source of truth
├── AGENTS.md               ← Auto-loaded by Claude, brief reference
├── CLAUDE.md               ← Claude Code guidance (shorter)
├── README.md               ← Human-facing project overview
├── requirements.txt
│
├── _MEMORY/                ← Per-session agent notes (YYYYMMDDHHMMSS.md)
├── _DOC/_PRD/              ← Product requirement docs
├── _DOC/_TEST/             ← Test specifications
├── _ARCH/                  ← Archived experiments, old notebooks
├── _BAK/                   ← Backup files (scripts, configs, parquets)
├── _LOG/                   ← ALL execution logs (pipeline, training, fetch)
│
├── pipeline/               ← Data infrastructure (shared)
│   ├── fetch/              ← L0 data acquisition
│   ├── feature/            ← L1 feature engineering
│   ├── plot/               ← Monte Carlo, P&L visualization
│   └── run/                ← Cron shell scripts
│
├── data/                   ← Data lake (shared, 3-layer)
│   ├── Level_0_Raw/
│   ├── Level_1_Features/
│   │   └── modules/        ← Modular feature parquets (7 files, 252 features)
│   └── Level_2_Datamart/
│
├── edges/                  ← Strategy research (TRAINING PRODUCT)
│   ├── bsjp_overnight_sl2/ ← ACTIVE — BSJP overnight
│   │   ├── edge.md         ← Strategy spec, iterations, lessons
│   │   ├── scripts/
│   │   │   ├── generate_datamart.py
│   │   │   ├── train_lightgbm.py
│   │   │   ├── build_close10_objective_datamart.py
│   │   │   ├── _TEMPLATE_generate_feature_module.py
│   │   │   ├── grid_search_coarse.sh
│   │   │   └── grid_search_fine.sh
│   │   └── analysis/
│   └── bpjs_opening_tp3/   ← PAUSED
│       ├── edge.md
│       ├── scripts/
│       └── analysis/
│
├── model/                  ← Trained models (TRAINING PRODUCT OUTPUT)
│   ├── BSJP/
│   │   ├── LATEST.md       ← Narrative: every model iteration, decisions, forensics
│   │   ├── bsjp_v7/        ← Reference baseline (clean, AUC 0.602)
│   │   ├── bsjp_v15/       ← CURRENT ACTIVE (clean, AUC 0.602, MaxDD -15.3%)
│   │   ├── bsjp_v10_*/     ← Close10 variants (LEAKAGE — invalidated)
│   │   ├── bsjp_grid_*/    ← Grid search results (LEAKAGE — invalidated)
│   │   └── _ARCH/          ← All other archived runs
│   └── BPJS/               ← Paused strategy
│
├── inferences/             ← Production inference (INFERENCE PRODUCT)
│   ├── PLANNING.md         ← Full inference PRD + feature spec
│   ├── _MEMORY.md          ← Inference-specific session notes
│   ├── bsjp/
│   │   ├── PERFORMANCE.md  ← Go vs Rust vs Python benchmarks
│   │   ├── PRD.md          ← Rust port PRD (blocked)
│   │   ├── db/             ← DuckDB database files
│   │   ├── python/         ← Python inference (ACTIVE production path)
│   │   │   ├── config.py
│   │   │   ├── db.py
│   │   │   ├── fetch.py           ← Production fetch (reads L2, upserts DuckDB)
│   │   │   ├── fetch_lightweight.py ← In-progress L0→DuckDB (merge bug)
│   │   │   ├── run.py             ← Model scoring + top-k picks
│   │   │   ├── bootstrap_feature_store.py
│   │   │   └── variants/
│   │   │       ├── v15.py         ← Current active variant
│   │   │       ├── v12.py
│   │   │       └── v12_guard.py
│   │   └── golang/         ← Go inference binary (FUTURE: momentum only working)
│   │       ├── cmd/bsjp/main.go
│   │       └── internal/
│   └── bpjs/               ← Paused inference
│
└── .claude/                ← Claude Code config
    ├── settings.local.json
    └── scheduled_tasks.lock
```

---

## 4. Current State (Ratified Decisions)

### Active, Clean, Production-Ready

| Component | Version | Status | Key Metric |
|-----------|---------|--------|------------|
| **Training model** | BSJP v17 | ✅ Reference | AUC 0.642, CumRet +3.44%/day, 305 trees, 100d OOT |
| **Training model** | BSJP v18_fixed | ✅ Retrain | AUC 0.67, 28 trees, 100d OOT, fixed-L2 only |
| **Training datamart** | `training_datamart_bsjp_overnight.parquet` (OLD) | ✅ CLEAN | 108,698 rows, pre-27-Apr labels |
| **Training datamart** | `training_datamart_bsjp_overnight.parquet` (NEW) | ⚠️ DIVERGED | 110,236 rows, post-27-Apr labels differ |
| **Training datamart** | `training_datamart_bsjp_overnight_fixed.parquet` | ✅ CARA PAKE | NEW rows + OLD core labels, 110,236 rows |
| **Feature modules** | 7 modules in `modules/` | ✅ ACTIVE | 252 features, PROVEN byte-identical OLD vs NEW |
| **Inference (Python)** | `fetch.py` + `run.py --variant v15` | ✅ PRODUCTION | 3.35s fetch + 2.58s score |
| **Inference (Go)** | `bsjp` binary | 🟡 PARTIAL | Tree-walk byte-identical ✅, CGO broker bug, yf_daily broken |
| **Strategy** | BSJP overnight | ✅ ACTIVE | Entry close 15:xx, exit open 09:xx T+1 |

### Suspended / Needs Attention

| Component | Status | Reason |
|-----------|--------|--------|
| BPJS strategy | ⏸️ PAUSED | 1h window too narrow to clear break-even |
| Python `fetch_lightweight.py` | 🟡 BLOCKED | Duplicate 'date' column bug after bucket rolling merge |
| Rust inference port | 🔴 BLOCKED | Polars first compile > 5 min, lightgbm-rs needs C headers |
| Go `broker_agg.go` CGO UPDATE | 🔴 BUG | RowsAffected=0 after UPDATE FROM, data IS written (verified Python) |
| Go `yf_daily.go` | 🔴 BROKEN | 58 OHLCV features produce wrong values |
| Go `overnight.go` | 🟡 CALIBRATE | gapdown_freq epsilon mismatch (0.00 vs 0.20) |
| Go CVD, Stockbit/XL, VWAP | 🔴 TODO | Not implemented yet (~16 features) |

### v18 Label Divergence — Root Cause (Apr 2026)

`generate_datamart.py` diubah 27 Apr 12:48 — mengubah cara hitung `entry_price`/`exit_price` → `overnight_return` berubah → `label_tp`/`label_sl2` berubah.

**Dampak:**
- OLD L2 (108,698 rows): `overnight_return` mean=0.000017, label_tp=34.6% → 768 trees, AUC 0.68
- NEW L2 (110,236 rows): `overnight_return` mean=-0.001470, label_tp=36.9% → 27 trees, AUC 0.62
- Common 105,102 rows have different labels (596 rows different label_tp, 592 different label_sl2)
- 5,134 extra rows in NEW (mostly 2023 dates, easier labels)
- 3,596 rows removed from OLD

**Fix:** `training_datamart_bsjp_overnight_fixed.parquet` — NEW row universe + OLD core labels (8 columns: date, ticker, entry_price, exit_price, overnight_return, label_tp, label_sl2, label_name). 110,236 rows, 28 trees, AUC 0.67.

**Lesson:** Versi generate_datamart.py dan train_lightgbm.py HARUS diversion-lock. Saat ini kedua script asli (pre-27-Apr) HILANG — tidak bisa mereproduksi v17/v18_BACKUP.

### Contaminated — INVALID (Data Leakage)

| Model / Datamart | Contamination | Status |
|------------------|---------------|--------|
| `bsjp_grid_fine/*` | `close_ret_last1h` leaked as feature in close10_v10 | ❌ INVALID — re-run needed |
| `bsjp_grid_coarse/*` | Same close10_v10 parquet | ❌ INVALID |
| `bsjp_v10_close10_obj` | close10 v1 parquet (pending investigation) | ⚠️ SUSPECT |
| `bsjp_v10_close10_obj_k2_w25_guard` | Same close10 v1 parquet | ⚠️ SUSPECT |
| `bpjs_v13` | Confirmed leakage, archived to `_ARCH/` | ✅ Handled |

### Ratified Engineering Decisions

1. **LightGBM binary classifier,** not LGBMRanker (v9d failed).
2. **Feature modules are the preferred fast path** for training — drop parquet to add features.
3. **Go is the future inference runtime** — 13 MB binary, no Python, no pip. But Python `fetch.py` + `run.py` remains production until Go achieves feature parity.
4. **DuckDB is the inference feature store** — not parquet, not SQLite.
5. **`min_data_in_leaf=100`, `lambda=1.0-1.5`** is the sweet spot (grid search finding). v7 default (md=500) is over-regularized.
6. **IHSG MA features work** — they provide macro context that lowers MaxDD without hurting AUC.
7. **Cross-sectional features (sq_, xc_, yp_, pd_)** are in the model but have NOT been audited for lookahead safety.
8. **Fixed L2 (`*_fixed.parquet`)** is the training artifact for v18+. NEW row universe (110,236) with OLD core labels. Regular rebuild (`generate_datamart.py`) untuk menambah tanggal baru; setelah rebuild, patch lagi 8 core columns dari backup.
9. **Training scripts MUST be version-locked** — `generate_datamart.py` dan `train_lightgbm.py` pre-27-Apr HILANG dan tidak bisa direproduksi.

### Next Priorities (Agreed)

1. **Version-lock training scripts** — `git init`, simpan checksum `generate_datamart.py` dan `train_lightgbm.py` yang verified. Setiap perubahan = git commit wajib.
2. **Paper trade v18_fixed** — run live, track results daily.
3. **Fix Go feature parity** — yf_daily (58 cols), CVD (6), Stockbit/XL (4), VWAP (6), overnight calibration, CGO broker UPDATE bug.
4. **Re-run grid search with fixed L2** — `training_datamart_bsjp_overnight_fixed.parquet`.
5. **Fix Python `fetch_lightweight.py`** duplicate column bug — zero-L1/L2 inference.
6. **Implement `generate_datamart.py` rebuild + patch workflow** — add new dates from L1, then overlay OLD 8 core columns from backup.

---

## 5. Engineering Standards

### Code Conventions

- Python 3.12, `pandas>=2.0`, `pyarrow>=14.0`
- Parquet for all persistence (no CSV, no JSON for structured data)
- `float32` dtype for all feature columns (consistent between training and inference)
- Shell scripts in `pipeline/run/` — one script per cron job
- Go code in `inferences/bsjp/golang/` — `go build -o bsjp ./cmd/bsjp/`

### Logging

- **ALL logs go to `_LOG/`** — not `edges/.../scripts/`, not `inferences/.../`
- Format: `{component}_{date}.log` or `{component}_{version}.log`
- Key log files:
  - `_LOG/broksum_resume_state.json` — broker fetch resume state
  - `_LOG/log_broksum.jsonc` — structured broker fetch log
- DuckDB `picks_log` table for production inference picks

### Model Versioning

- Model output: `model/BSJP/bsjp_vN/` (N = sequential integer)
- Variants within a version: `bsjp_vN_variant/` (e.g., `bsjp_v15_1` for repro)
- Each model directory must contain: `metrics.json`, `model_lightgbm_*.txt`, `feature_importance.csv`
- After training, ALWAYS update `model/BSJP/LATEST.md` with results
- Never delete old models — move to `model/BSJP/_ARCH/`

### Path References

- **Use absolute paths or repo-root-relative paths only** in code
- **NO hardcoded `/home-ssd/` paths** — these drift across machines
- Reference paths from repo root: `idx/data/Level_0_Raw/`, `idx/pipeline/`, etc.
- Shell scripts use `$IDX_DIR` variable set at top

### Testing

- Before trusting any model: verify feature count, AUC range, overfit gap
- Before deploying inference: compare 5 sample dates byte-to-byte vs training pipeline
- Monte Carlo mandatory for any model considered "production-ready": 10K paths, block bootstrap (block=5)

---

## 6. Session Protocol

### Start Ritual (Agent Reads)

1. `program.md` — this file, always first
2. `model/BSJP/LATEST.md` — what models exist, what's current, what's broken
3. `_MEMORY/` — find latest session file, read it
4. Identify which product you're working on. Explicitly state: "Working on PRODUCT: Training" or "Working on PRODUCT: Inference"

### Work Ritual (Agent Does)

- If TRAINING: work only in `edges/`, `model/`, `data/Level_2_Datamart/`
- If INFERENCE: work only in `inferences/`, `pipeline/run/run_inference_bsjp.sh`
- SHARED changes (data/Level_0 or Level_1) — flag as potentially affecting both products
- Never run L1/L2 rebuild from inference code
- Never read DuckDB from training code

### End Ritual (Agent Writes)

1. Create `_MEMORY/YYYYMMDDHHMMSS.md` with:
   - Date, scope
   - Key decisions made
   - Major changes implemented (file paths, line counts)
   - Data/state findings
   - Current production state snapshot
   - Next session priority (explicit handoff)
2. If model iterations changed: update `model/BSJP/LATEST.md`
3. If product boundaries changed: update this `program.md`
4. Do NOT leave TODO comments in code — use `_MEMORY/` for handoff

---

## 7. Quick Reference

### Training a New Model

```bash
source ../.venv/bin/activate
bash pipeline/run/run_fetch_broksum.sh
bash pipeline/run/run_feature_l1.sh
cd edges/bsjp_overnight_sl2/scripts
python generate_datamart.py --strategy-mode bsjp
# ⚠️ Rebuild changes core labels — patch 8 columns from backup (see §4 v18 Label Divergence)
python train_lightgbm.py \
  --output-dir ../../model/BSJP/bsjp_vN \
  --feature-modules-dir ../../data/Level_1_Features/modules \
  --feature-prune-top-n 0 \
  --tp-pct 0.01 --sl-pct -0.02 \
  --oot-valid-days 100 \
  --min-data-in-leaf 100 \
  --lambda-l1 1.0 --lambda-l2 1.5
```

### Running Daily Inference

```bash
# Bootstrap (one-time):
python inferences/bsjp/python/bootstrap_feature_store.py --replace

# Daily (cron):
python inferences/bsjp/python/fetch.py
python inferences/bsjp/python/run.py --variant v15 --log-picks
```

### Go Binary (Future)

```bash
cd inferences/bsjp/golang
go build -o bsjp ./cmd/bsjp/
./bsjp bootstrap
./bsjp fetch && ./bsjp predict --variant v15 --log
```

---

## Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2026-04-29 | 1.0 | Initial program.md. Defined two-product architecture, invariants, directory structure, session protocol. Based on AGENTS.md, LATEST.md, PLANNING.md, and existing codebase state. |
| 2026-04-29 | 1.1 | Post v18 investigation: Added L2 label divergence root cause, fixed datamart, Go parity bugs, script version-loss warning, updated training params (md=100, λ=1.0/1.5), revised priorities. |
