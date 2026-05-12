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
│   ├── run/                ← Cron shell scripts
│   ├── storage/            ← L0 DuckDB abstraction (PRD 0003, Phase 0 done)
│   │   ├── config.py       ← L0_SOURCES + canary feature flags
│   │   ├── l0.py           ← L0_DB_FILES, attach_l0(), open_l0_writer()
│   │   ├── writers.py      ← write_l0() flag-driven dual-write
│   │   ├── readers.py      ← read_l0() with duckdb_read toggle
│   │   ├── backup.py       ← 3-tier retention (weekly/monthly/yearly)
│   │   ├── migrate.py      ← init_all() + CLI
│   │   ├── schemas/        ← TableSchema per source (master_emiten only)
│   │   ├── validators/     ← BaseValidator + per-dtype tolerance
│   │   └── tests/          ← Continuity merge gate (pytest)
│   └── migrations/         ← Phase migrations (0003_phase0_init done)
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

### Current BSJP Research Handoff (2026-05-09)

This is the current **Training/research** state. It supersedes the older v18/v19/v20 headline tables below for any new BSJP research decision.

| Item | Current State |
|---|---|
| Current clean candidate | `model/BSJP/v23b_t1audit2_clean/` |
| Objective | BSJP `close10`: entry close 15:xx T, exit open 10:xx T+1 |
| Status | **PROMOTED TO PRODUCTION INFERENCE (2026-05-09).** Go parity updated with `ara_history`. |
| OOT artifact window | `valid_predictions.parquet`: 100 trading days, 2025-11-17 -> 2026-04-23 |
| Latest calendar extension | Provisional local scoring to latest closed close10 date 2026-05-06 |
| Reason latest closed date is not 2026-05-09 | 2026-05-09 is Saturday; 2026-05-08 entry has no T+1 exit yet; local 1h file also lacks the 2026-05-08 10:xx candle needed to close 2026-05-07 |
| Locked OOT metrics | AUC 0.5346, cum net +207.9%, MaxDD -24.2%, best iteration 5, overfit gap 0.0616 |
| Current quick-win policy candidate | k=2, max weight 25%, cost cap 3%, adaptive q=.85 |
| Quick-win locked OOT | +255.0%, MaxDD -17.9%, active days unchanged at 96 |
| Conservative policy candidate | k=2, max weight 20%, q=.90: +184.7%, MaxDD -13.5% |

No-lookahead status for `v23b_t1audit2_clean`:

- Major same-day broker leakage from v19d/v20 has been patched.
- Residual CVD path was patched to T-1.
- Residual `yf_daily_*` aggregate leakage was patched by shifting all raw current-row L1 numeric families before aggregate/recompute.
- `ara_history_features.parquet` was regenerated against current `yfinance_daily`.
- Final audit artifact: `_LOG/v23b_t1audit2_clean_no_lookahead_audit_20260509.json`.
- Hard audit failures are all false: no feature blacklist leak, no outcome leak, no policy-only leak, pre14 cutoff max hour 14, and broker shifted aggregate exact-match checks pass.

Feature contribution for the current candidate:

| Family | Gain share | Read |
|---|---:|---|
| pre14 intraday | ~67.1% | Main signal source |
| macro prev-close | ~26.7% | IHSG/USDIDR/VIX/global context |
| ARA-history T-1 | ~2.6% | Small helper, not pure ARA hunter |
| `yf_daily` T-1-or-older | ~1.8% | Safe after T-1 audit patch |
| broker aggregate T-1 | ~1.6% | Safe but weak after leak patch |
| CVD T-1 | ~0.07% | Negligible |

Interpretation in plain language:

- Old v19d/v20 headline returns are not credible because they were inflated by same-day broker leakage and ARA fillability fantasy.
- Current v23 is not a pure ARA hunter. It is mostly a pre-14 intraday + macro regime model, with small help from ARA-history and clean T-1 broker context.
- Simple train-time feature pruning was tested and should not be pursued now: `v23d_pruned78_t1audit_clean` and `v23e_pruned50_t1audit_clean` did not improve portfolio quality.
- Remaining risk is **robustness/overfit**, not a confirmed leakage failure in the current audit scope. Best iteration is still very low (5), so full rolling-retrain validation is mandatory before promotion.

Latest Rp10m calendar extension, using quick-win k=2/w25 and latest locally computable closed close10 date 2026-05-06:

| Window | Period | Ending Capital | PnL | Return |
|---|---|---:|---:|---:|
| 7 trading days | 2026-04-27 -> 2026-05-06 | Rp11,131,121 | +Rp1,131,121 | +11.31% |
| 30 trading days | 2026-03-17 -> 2026-05-06 | Rp12,247,629 | +Rp2,247,629 | +22.48% |
| 90 trading days | 2025-12-11 -> 2026-05-06 | Rp27,150,569 | +Rp17,150,569 | +171.51% |

Important caveat: the post-2026-04-23 calendar extension is provisional. It is not the locked OOT artifact, and broker aggregate/CVD module coverage currently ends at 2026-04-23, so post-OOT rows have missing broker-family values.

### Active, Clean, Production-Ready

Historical table. Use the handoff above for current BSJP research state.

| Component | Version | Status | Key Metric |
|-----------|---------|--------|------------|
| **Training model** | BSJP v17 | ✅ Reference | AUC 0.642, CumRet +3.44%/day, 305 trees, 100d OOT |
| **Training model** | BSJP v18_fixed | ✅ Retrain | AUC 0.67, 28 trees, 100d OOT, fixed-L2 only |
| **Training model** | BSJP `v18_close10_rebuild` | 🟡 Model | AUC 0.650, 408 trees, raw k3/w34 MaxDD -35.6% |
| **Paper-trade policy** | `v18_close10_rebuild_policy_w25` | 🟡 Candidate | k=3, max weight 25%, OOT +0.98%/day, MaxDD -28.2%, MC 100d P(loss)=1.78% |
| **Training datamart** | OLD `training_datamart_bsjp_overnight.parquet` | ✅ v18 baseline | 108,698 rows, actually `bsjp_close10_sl2` (exit open@10), misleading filename |
| **Training datamart** | NEW `training_datamart_bsjp_overnight.parquet` | ✅ true overnight | 110,236 rows, `bsjp_overnight_sl2` (exit open@09) |
| **Training datamart** | `training_datamart_bsjp_close10_rebuild_v18like.parquet` | 🟡 v18-like | 108,698 rows, close10 label rebuilt on OLD row universe |
| **Training datamart** | `training_datamart_bsjp_overnight_fixed.parquet` | ⚠️ HYBRID | NEW rows + OLD core labels; useful for forensics, semantically mixed |
| **Feature modules** | `modules/*_features.parquet` | ✅ ACTIVE | Expanded module set; v23 uses 315 selected model features |
| **Inference (Python)** | archived path | 🗑️ ARCHIVED | Do not use for daily production |
| **Inference (Go)** | `bsjp` binary | ✅ ACTIVE | Current production path via `pipeline/run/run_inference_bsjp.sh`; v23 not handed off |
| **Strategy** | BSJP close10 | ✅ TARGET OBJECTIVE | Entry close 15:xx, exit open 10:xx T+1 |

### L0 → DuckDB Migration (PRD 0003, Phase 0 Done — Phase 1 Quick Win Landed)

| Item | Status |
|---|---|
| PRD 0003 (`_DOC/_PRD/0003_l0_to_duckdb.md`) | ✅ rev 0.5 — sequencing, canary pattern, feature flag state machine, continuity invariants, orphan source verdict |
| `git init` baseline + `.gitignore` | ✅ done (commit `ce04e9c6`, 158 files) |
| L0 parquet backup (`_BAK/L0_pre_duckdb_20260429/`) | ✅ done (193 MB, 10 files) |
| `duckdb==1.5.2` pinned in `requirements.txt` | ✅ done |
| **B1** — foundation library (`config.py`, `schemas/_base.py`) | ✅ done |
| **B2** — first concrete schema (`schemas/master_emiten.py`) | ✅ done (roundtrip 773 rows verified) |
| **B3** — dual-write engine (`l0.py`, `writers.py`, `readers.py`, `validators/_base.py`) | ✅ done (7 e2e tests pass) |
| **B5** — continuity merge gate (`tests/test_continuity.py`) | ✅ done (pytest 4/4 — auto-extended to master_broker) |
| **B4** — backup + migrate + `0003_phase0_init.py` | ✅ done (7 e2e tests pass; backup roundtrip via `IMPORT DATABASE` verified) |
| Run scripts (`run_backup_l0_duckdb.sh`, `run_validate_l0_duckdb.sh`) | ✅ done (live smoke test on stage 0; force-Sunday backup roundtrip clean) |
| Bootstrap real `master.duckdb` | ✅ done (1.3 MB) |
| **Phase 1 quick win**: `master_emiten` + `master_broker` schemas + initial bulk load via `0003_phase1_master.py` | ✅ done (773 + 92 rows; validator parity 0 diffs) |
| Refactor fetch scripts (`fetch_emiten.py`, `fetch_master_broker_idx.py`) → `write_l0()` | ⏳ next |
| Stage 1 promotion (`L0_MASTER_*_DUCKDB_WRITE=true`) + 14-day streak | ⏳ after fetch refactor |
| Stage 2 read switch + Stage 3 cutover | ⏳ Phase 1 close-out |
| Phase 5 broksum chunked benchmark | ⏳ defer to Phase 5 |

**Active migration invariant:** default canary flags = stage 0 (parquet only, no DuckDB writes/reads). All current pipelines unaffected until env vars promote individual sources. Continuity gate enforces this at merge time for any fetch script refactor.

**Storage abstraction is internally consistent** — a new schema dropped into `pipeline/storage/schemas/<source>.py` + registered in `schemas/__init__.py` is automatically picked up by writers, readers, validators, migration runner, and the continuity gate. Verified end-to-end on `master_broker` (auto-extended pytest gate from 2 → 4 tests with zero edits).

**Bugs caught + fixed via real-data Phase 1 load** (would have been missed by synthetic fixtures):
1. **DuckDB rejects `%` in unquoted identifiers** — `master_broker` has `foreignfund_%` etc. Fixed by `_q()` helper double-quoting all identifiers in `_base.py` SQL generators.
2. **Validator false-positive on TIMESTAMP cols** — parquet stores ISO strings, DuckDB returns `datetime64`; native `==` flags 100% diff. Fixed via `pd.to_datetime` normalization in `_compare_column`.

**Audit gap resolved (2026-05-01):** `ipot_ohlcv_1h.parquet` → **deferred** (PRD 0003 §11.1, rev 0.5). Zero live consumer found (former Python inference path archived; Go path doesn't reference it). Fetcher kept; no L0 schema, no migration. Flagged separately: `run_inference_bsjp.sh` chain is broken (Suspended table below).

### Active BSJP Research Baselines (May 2026)

The active research framing is now split by execution/fillability regime. `v18_close10_rebuild_policy_w25` is retained only as a historical recovered-v18 reference because it depends on close15/EOD-style features that are not available before a realistic 14:59 decision.

| Baseline | Artifact | Use | 100D | 50D | 20D | MaxDD | Status |
|---|---|---|---:|---:|---:|---:|---|
| **ARA continuation** | `model/BSJP/bsjp_v19d_close10_preclose14_orb_md100_l21.5/` | Near-ARA/ARA continuation research; needs fillability realism | +148.6% | +24.3% | +9.9% | -29.0% | Active research |
| **ARA cont. policy** | `model/BSJP/bsjp_v20_ara_continuation_state_policy_clean/` | v20 clean veto: keep single-release + near-ARA (0-3%), veto rest | +222.6% | +66.1% | +23.0% | -3.2% | Policy artifact, implemented in Go predict |
| **Non-ARA continuation** | `model/BSJP/bsjp_v19d_close10_preclose14_orb_md100_l21.5_noara/` | Control baseline after hard excluding near-ARA names | -42.3% | n/a | n/a | -44.2% | Negative control, not tradable |

Key finding: hard excluding `pre14_is_ara_like` removes the current edge. The no-ARA baseline is therefore not a promotion candidate; it is the control proving that the v19d edge is concentrated in difficult-to-fill ARA-like names. Next research should make the ARA-continuation branch fillability-aware, and separately search for a better non-ARA continuation signal.

ARA-state diagnostic now splits the ARA branch more realistically:

| Bucket | Initial Read |
|---|---|
| `ara_touched_single_release` | Strongest selected bucket so far; release wick implies some fillability |
| `ara_touched_repeated_release` | Much weaker; repeated release wicks may indicate inventory distribution |
| `near_ara_not_touched_0_3pct` | Still tradable; small sample but strong |
| `non_ara_far_gt8pct` | Weak current continuation baseline |

Artifacts: `_LOG/pre14_ara_state_bucket_diagnostic_20260501.csv` and `_LOG/pre14_ara_state_selected_examples_20260501.csv`.

Policy simulation without retrain:

| Policy | 100D | 50D | 20D | MaxDD | Read |
|---|---:|---:|---:|---:|---|
| `veto_keep_single_near_momentum_0_8` | +248.6% | +59.0% | +18.7% | -3.2% | Best veto-mode diagnostic |
| `veto_keep_single_plus_near_0_3` | +222.7% | +66.1% | +23.0% | -3.2% | Cleaner ARA-fillability branch |
| `veto_repeated_only` | -9.2% | -15.8% | -8.3% | -20.6% | Repeated release weak |
| `veto_far_nonara_only` | -24.8% | -11.3% | +0.9% | -32.0% | Far non-ARA weak |

Artifacts: `_LOG/pre14_ara_state_policy_sim_summary_20260501.csv` and `_LOG/pre14_ara_state_policy_veto_summary_20260501.csv`. Treat this as OOT diagnostic policy selection, not promotion; next step is frozen-rule validation.

Frozen walk-forward validation over the original 4 pre-OOT folds:

| Policy | 80d WF CumNet | MaxDD | Net/Trade | Read |
|---|---:|---:|---:|---|
| `wf_all_base` | +174.1% | -13.8% | +1.99% | Highest total return |
| `wf_veto_single_near_momentum_0_8` | +111.4% | -7.4% | +6.12% | Positive all folds; better risk/trade quality |
| `wf_veto_single_plus_near_0_3` | +92.6% | -7.4% | +6.09% | Cleaner fillability branch; positive all folds |
| `wf_veto_far_nonara_only` | -16.2% | -22.3% | -0.47% | Weak branch |

Artifacts: `_LOG/pre14_ara_state_policy_walkforward_summary_20260501.csv` and `_LOG/pre14_ara_state_policy_walkforward_by_fold_20260501.csv`. Read: ARA-state veto validates as a risk filter, but not yet as a return upgrade versus full base in walk-forward.

Packaged baseline artifact:

- `model/BSJP/bsjp_v20_ara_continuation_state_policy_clean/`
- Policy: keep `ara_touched_single_release` + `near_ara_not_touched_0_3pct`
- Mechanic: veto after original v19d selection, no re-ranking/reweighting
- OOT: 100D +222.6%, 50D +66.1%, 20D +23.0%, MaxDD -3.2%, 48 positions
- Status: research baseline artifact, not production

Next branch: **v20 no-touch clean**. Rationale: even single-release ARA touched may be hard to fill live; the most operationally realistic seed is `near_ara_not_touched_0_3pct`. Initial no-touch diagnostic:

| Bucket | OOT 100D | WF Validation | Read |
|---|---:|---:|---|
| `near_ara_not_touched_0_3pct` | +9.6% | +17.4% | cleanest but sparse |
| `near_momentum_0_8_not_touched` | +18.6% | +28.7% | broader candidate, needs filters |
| `all_not_touched` | -11.0% | +7.5% | too noisy |

Artifact: `_LOG/pre14_ara_state_no_touch_alternatives_20260501.csv`. Next session should start from this no-touch branch and search for filters that increase signal count without admitting ARA-touched names.

### Suspended / Needs Attention

| Component | Status | Reason |
|-----------|--------|--------|
| BPJS strategy | ⏸️ PAUSED | 1h window too narrow to clear break-even |
| Python `fetch_lightweight.py` | ⏸️ DEFERRED | Duplicate 'date' column bug; Go inference replaces it |
| Rust inference port | 🔴 BLOCKED | Polars first compile > 5 min, lightgbm-rs needs C headers |
| Go `broker_agg.go` CGO UPDATE | 🟡 COSMETIC | RowsAffected=0 after UPDATE FROM, data IS written (base flow 18/18 PERFECT) |
| Go broker timeflow/context | 🟡 DEFERRED | 96 cols rely on bootstrap values; not regenerated in Go. Phase 2-3 needed. |
| `ipot_ohlcv_1h.parquet` | ⏸️ DEAD CODE | Zero live consumers (PRD 0003 §11.1). Removed from inference cron. Fetcher still runs. |
| Python inference path | 🗑️ ARCHIVED | `inferences/bsjp/python/` removed. Go binary is now sole inference path. |
| `run_inference_bsjp.sh` | ✅ FIXED | Now calls Go binary: `bsjp fetch` + `bsjp predict`. |

### v18 Label Divergence — Corrected Root Cause (Apr 2026)

The original diagnosis was "label drift after `generate_datamart.py` changed." The corrected diagnosis is more specific: **OLD v18 used close10 labels even though the file was named `training_datamart_bsjp_overnight.parquet`**.

OLD v18 semantics:
- Entry = `close@15 T`
- Exit = `open@10 T+1`
- `label_name = bsjp_close10_sl2`

NEW rebuilt overnight semantics:
- Entry = `close@15 T`
- Exit = `open@09 T+1`
- `label_name = bsjp_overnight_sl2`

**Dampak:**
- `entry_price` did not drift on common OLD vs NEW rows.
- `exit_price` changed on 85,945 / 105,102 common rows (81.8%) because exit hour changed 10 → 9.
- `label_tp` changed on 32,967 / 105,102 common rows (31.4%).
- `label_sl2` changed on 19,202 / 105,102 common rows (18.3%).
- This objective mismatch explains the tree collapse from 305-ish trees to 27/28 trees.

**Recovered path:** Rebuild with current `generate_datamart.py --exit-hour 10`, then filter to the OLD `(date,ticker)` universe:
- Full close10 rebuild: `data/Level_2_Datamart/training_datamart_bsjp_close10_rebuild.parquet` (424,909 rows; too broad for v18 reproduction).
- v18-like close10 rebuild: `data/Level_2_Datamart/training_datamart_bsjp_close10_rebuild_v18like.parquet` (108,698 rows; core labels match OLD except 1 row).
- Model: `model/BSJP/bsjp_v18_close10_rebuild/` → 408 trees, OOT AUC 0.650, raw k3/w34 mean daily net +1.30%, MaxDD -35.6%.
- Recommended paper-trade policy: `model/BSJP/bsjp_v18_close10_rebuild_policy_w25/` → same model, k=3, max weight 25%, OOT mean daily net +0.98%, MaxDD -28.2%, worst day -7.6%.
- Monte Carlo for policy w25: 10,000 paths, block=5. 100d median 2.57x, P(loss)=1.78%, mean MaxDD -19.6%, P(MaxDD≤-30%)=7.61%. 252d median 11.34x, P(loss)=0.04%, mean MaxDD -25.2%, P(MaxDD≤-30%)=22.0%.

**Lesson:** Names must encode objective. Do not call a close10 datamart `overnight`. Training scripts and datamart artifacts must be version-locked with checksums and exact commands.

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
8. **Close10 v18-like L2 is the current recovered training artifact** for reproducing v18 behavior: `training_datamart_bsjp_close10_rebuild_v18like.parquet`. The `*_fixed.parquet` file is a mixed forensic artifact, not clean canonical training data.
9. **Training scripts MUST be version-locked** — `generate_datamart.py` dan `train_lightgbm.py` pre-27-Apr HILANG dan tidak bisa direproduksi.

### Next Priorities (Agreed)

1. **Run proper rolling-retrain validation for `v23b_t1audit2_clean`** — do not promote based only on final OOT/subwindow diagnostics.
2. **Decide policy after rolling validation** — current candidates are quick-win k=2/w25/q85 and conservative k=2/w20/q90.
3. **Optimize full policy grid simulator** — current micro-grid works; full pandas loop is too slow.
4. **Run ablations** — no-broker, no-macro, no-pre14, and fold-stable feature contribution to understand whether v23 is robust or fragile.
5. **Close post-2026-04-23 coverage gap** — broker aggregate/CVD modules currently end at 2026-04-23; calendar extension after that has missing broker-family values.
6. **Only after research acceptance, plan inference handoff** — v23 feature list/policy are not yet production Go parity work.

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
python generate_datamart.py --exit-hour 10
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
# Daily (cron):
bash pipeline/run/run_inference_bsjp.sh
```

### Go Binary

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
| 2026-04-30 | 1.2 | Corrected v18 root cause: OLD datamart was close10 despite overnight filename. Added v18 close10 rebuild, MC results, and k3/w25 paper-trade policy. |
| 2026-04-30 | 1.3 | Added `current_v18_fixed` forensic conclusion: hybrid close10/overnight labels are not promotable; clean fixed-universe close10 test failed. |
| 2026-04-30 | 1.4 | Added v19 preclose14 cost-aware baseline: modular pre-14:59 features, close15/EOD blacklist preset, and cost-aware labels. |
| 2026-04-30 | 1.5 | Added v19b gross-label preclose14 filter tests: gross alpha without close15/EOD is weak; cost/price filters help but are not promotable. |
| 2026-04-30 | 1.6 | Added v19c preclose14 volume/turnover expansion and 9-run hyperparameter sanity grid; `md100_l2=1.5` is a research candidate only. |
| 2026-04-30 | 1.7 | Added v19d ORB preclose14 proxy results; first executable proxy family to materially improve 100d/50d/20d, still research only. |
| 2026-04-30 | 1.8 | Added §4 "L0 → DuckDB Migration" subsection (PRD 0003 progress). Added `pipeline/storage/` to §3 directory tree. B1 (foundation library) + B2 (master_emiten schema) done. |
| 2026-04-30 | 1.10 | Phase 0 storage library complete. B3 (writers/readers/l0/validators), B5 (continuity merge gate via pytest), B4 (backup 3-tier + migrate + `0003_phase0_init.py`) all landed and verified. Expanded §3 dir tree + §4 L0 migration table. Bootstrap of real `master.duckdb` still pending user approval. |
| 2026-05-01 | 1.11 | Bootstrapped real `master.duckdb` (empty table, 274 KB, gitignored). Audit `ipot_ohlcv_1h.parquet` → deferred (zero live consumer; PRD 0003 §11.1 rev 0.5). Added `run_inference_bsjp.sh` to Suspended table (Step 2 path broken — references archived `inferences/bsjp/python/fetch.py`). |
| 2026-04-30 | 1.9 | Added v19d ARA-like execution-filter finding: hard excluding near-ARA names turns v19d negative, proving the current ORB edge is concentrated in difficult-to-fill ARA-like picks. |
| 2026-05-01 | 1.12 | Reframed active BSJP research baselines as ARA-continuation vs non-ARA continuation; v18 is historical recovered reference only, not an active baseline. |
| 2026-05-01 | 1.13 | Added ARA-state/release-wick diagnostic: single release wick bucket is strong, repeated release wick bucket is weak, supporting a fillability-vs-distribution split. |
| 2026-05-01 | 1.14 | Added ARA-state policy simulation without retrain; veto-mode keeping single-release/near-ARA buckets improves PnL and drawdown, but requires frozen-rule validation. |
| 2026-05-01 | 1.15 | Added frozen ARA-state walk-forward validation: single/near buckets are positive in all folds and improve drawdown/trade quality, but full base still has higher pre-OOT cumulative return. |
| 2026-05-01 | 1.16 | Packaged `bsjp_v20_ara_continuation_state_policy_clean` policy-layer artifact with metrics, trades, daily PnL, charts, and source v19d model copy. |
| 2026-05-01 | 1.17 | Added no-touch ARA continuation handoff: start next from near-ARA-not-touched branch because it is more operationally fillable than ARA-touched single-release. |
| 2026-05-01 | 1.18 | L0 migration: run scripts landed (`run_backup_l0_duckdb.sh`, `run_validate_l0_duckdb.sh`). Phase 1 quick win — `master_broker` schema authored, `0003_phase1_master.py` migration ran, `master.duckdb` populated with 773 emiten + 92 broker rows. Validator parity 0 diffs both tables. Two bugs caught via real-data run + fixed: DuckDB `%` quoting, validator timestamp normalization. Continuity gate auto-extended to 4 tests. |
| 2026-05-04 | 1.19 | Go inference: CVD + Preclose14 implemented (360 cols/ticker for v19d). yf_daily fixed (`.JK` suffix). DuckDB `INSERT OR REPLACE` dup-key fix. Broker/CVD flow order fix. Calibration: Global (15/15), Momentum (4/4), yf_daily (58/58) PERFECT. v20 ARA-state policy (`v20_clean`) implemented in Go predict path — auto-active for v19d/v20 variants. Preclose14 volume calibration (~30 cols diff) and overnight epsilon still pending. |
| 2026-05-04 | 1.20 | Go inference parity: Overnight rewritten to daily parquet (24/27 PERFECT). Stockbit/XL implemented (4 cols, PERFECT). Preclose14 volume fixed (min_periods bug). Broker base flow calibrated (18/18 PERFECT vs fresh Python — previous diff was stale module data). `run_inference_bsjp.sh` replaced with Go binary. Python inference path (`inferences/bsjp/python/`) archived. `bsjp download` command added (Yahoo Finance 1h fetcher). All Go Parity Gaps resolved except broker timeflow/context (96 cols, Phase 2-3 deferred). |
| 2026-05-09 | 1.21 | Added current v23 handoff: `v23b_t1audit2_clean` is the clean research candidate after broker/CVD/yf_daily/ARA-history audits; k=2/w25/q85 is quick-win policy candidate; latest provisional closed PnL ends 2026-05-06, not 2026-05-09; next gate is full rolling-retrain validation before inference handoff. |
| 2026-05-10 | 1.22 | **Go Parity Breakthrough:** Implemented Hybrid Feature Loading in Go. Engine now loads complex Z-scores and Macro features from research parquets while computing real-time aggregates from L0. Fixed major parity gap for v23b (315/315 features now accessible in Go). Refactored `broker_agg.go` to include Bandar/LocalFund groupings. |
| 2026-05-10 | 1.23 | **Super Deep Dive Audit:** Scanned all feature modules (10+). Fixed 2.6M duplicates in Sector features. Discovered and fixed 95% NaN bug in Closing Momentum via session-time normalization. Harmonized v25 features (Forensic V2 + Sector) with 100% logic parity. Locked high-integrity datamart for v25 training. |
inference handoff. |
