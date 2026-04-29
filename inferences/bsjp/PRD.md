# PRD: BSJP Inference Engine — Rust Port

| | |
|---|---|
| **Author** | Engineering |
| **Status** | Draft |
| **Created** | 2026-04-28 |
| **Target GA** | T+14 days |
| **Stakeholders** | Quant Research, DevOps |

---

## 1. Executive Summary

Replace the Python inference pipeline (`fetch_lightweight.py` + `run.py`) with a **single statically-linked Rust binary** that reads L0 parquet directly, computes 249 feature vectors, persists to DuckDB, and scores top-3 picks — **no Python runtime, no L1/L2 dependency, no pip install**.

### 1.1 Why

| Incident | Date | Impact |
|---|---|---|
| `fetch.py` freshness check triggered full L1+L2 rebuild | 2026-04-28 | 15 hours downtime, DuckDB stale 3 trading days |

Root cause: the Python pipeline conflates feature computation with pipeline maintenance. The Rust binary eliminates this by design — it never touches L1/L2 infra.

### 1.2 Success Criteria

| Metric | Target | How Measured |
|---|---|---|
| End-to-end runtime (fetch + predict) | < 5 minutes | wall-clock on VPS 4 GB |
| RAM ceiling | < 2 GB | peak RSS, `time -v` |
| Binary size | < 50 MB | `ls -lh` on release build |
| Feature MAE vs training pipeline (per column) | < 1e-10 | `assert_frame_equal` over 5 sample dates |
| DuckDB backfill (2 years, 730 dates) | < 30 seconds | bootstrap mode |
| Missing days (30-day rolling) | 0 | `SELECT MAX(date) GAP` from `features_store` |
| Training files modified | 0 | `git diff --stat` on `pipeline/`, `data/Level_1_*`, `data/Level_2_*` |
| Cold-start deploy time (bare VPS) | < 10 minutes | `scp binary && ./bsjp bootstrap` |

---

## 2. Architecture

### 2.1 Data Flow

```
L0 Parquet (60-day rolling window, ~80 MB)
├── broksum_bybroker.parquet
├── yfinance_1h.parquet
├── yfinance_daily.parquet
├── global_indices.parquet
├── master_broker.parquet
└── master_emiten.parquet
        │
        ▼
┌──────────────────────────────────────────────────┐
│  bsjp (single binary)                            │
│                                                  │
│  Subcommands:                                    │
│  ├── fetch    L0 → features → DuckDB upsert      │
│  ├── predict  DuckDB → LightGBM → top-3 picks    │
│  └── bootstrap  backfill DuckDB from L2 parquet  │
│                                                  │
│  Modules:                                        │
│  ├── broker::features    flow/ctx/temporal       │
│  ├── broker::buckets     localfund, bandar        │
│  ├── broker::cross_sectional  xc_, sq_, yp_, pd_ │
│  ├── ohlcv::momentum     close_ret_last1h etc    │
│  ├── ohlcv::overnight    gap history 30 cols     │
│  ├── ohlcv::derived      yf_ return/volume/range │
│  ├── global::features    IHSG, VIX, USDIDR, etc  │
│  ├── cvd::features       volume accumulation     │
│  └── duckdb::store       schema, upsert, query   │
│                                                  │
│  CLI: clap (derive)                              │
│  Data: polars 0.53 (lazy, parquet)               │
│  DB:   duckdb-rs                                  │
│  ML:   lightgbm ffi OR treelite OR native port   │
└──────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│  inferences/bsjp/db/inference.duckdb             │
│  ├── features_store  (date, ticker, f1..f249)   │
│  └── picks_log       (date, variant, rank, ...)  │
└──────────────────────────────────────────────────┘
```

### 2.2 Subcommands

```bash
# Daily production (cron 17:30 WIB)
bsjp fetch                            # compute today → upsert
bsjp predict --variant v15 --log      # score → top-3 → log picks

# One-time bootstrap (or after model feature-list change)
bsjp bootstrap                        # L2 parquet → DuckDB backfill

# Utility
bsjp fetch --date 2026-04-24          # specific date
bsjp check                            # freshness report
bsjp calibrate --date 2026-04-23      # compare vs training pipeline output
```

### 2.3 Freshness Logic (fetch)

```
fetch(target_date):
  1. Query DuckDB: MAX(date) FROM features_store
  2. IF max_date >= target_date → exit 0 (noop)
  3. Load 60-day L0 window ending target_date
  4. IF broksum_bybroker has target_date:
       compute features for target_date → upsert → exit 0
     ELSE:
       warn "No broker data for {target_date}, using T-2"
       retry with target_date - 2
```

**Key invariant:** Never reads or writes L1/L2 parquet files. Never triggers Python scripts. If data is missing, skip and warn.

---

## 3. Technical Specification

### 3.1 Crate Stack

| Crate | Version | Purpose |
|---|---|---|
| `polars` | 0.53 | DataFrame, lazy queries, parquet I/O |
| `duckdb` | 1.x | Embedded OLAP feature store |
| `clap` | 4.x | CLI (derive API) |
| `anyhow` | 1.x | Error propagation |
| `thiserror` | 2.x | Typed error enums |
| `tracing` | 0.1 | Structured logging |
| `tracing-subscriber` | 0.3 | Log output (JSON for prod, pretty for dev) |
| `chrono` | 0.4 | Date arithmetic, WIB timezone |
| `serde` / `serde_json` | 1.x | Config deserialization |
| `lightgbm` ffi | — | Model inference (TBD: evaluate treelite, native port, or C FFI) |

### 3.2 Feature Pipeline (fetch)

All feature computation follows the L1→L2 transform spec in `inferences/PLANNING.md` §5.

#### 3.2.1 Look-Ahead Guard

Every transformation must enforce `shift(1)` on broker and OHLCV data:
- Broker features at T use T-1 broker summary data
- Overnight history at T uses rolling windows ending T-1
- Entry price (close@15:xx T) is the only T timestamp feature allowed

Validation: at runtime, assert no column is sourced from a date > target_date for broker features. Panic if violated (compile-time invariant is ideal but not easily enforced in Polars lazy API — prefer eager checks on the final 60-row window).

#### 3.2.2 Module Contracts

Each feature module exports:
```rust
pub fn compute(df: LazyFrame, config: &Config) -> PolarsResult<LazyFrame>;
```
Input grain: `(date, ticker)` for L2-level features, `(date, broker, ticker)` for L1-level.
Output: same grain + new columns with the module's prefix.

Modules are stacked via `left_join` on key columns.

#### 3.2.3 Feature Families

| Module | Prefix(es) | New Columns | Grain | Rolling Window |
|---|---|---|---|---|
| `broker::flow` | `flow_` | 18 | (date, ticker) | 60d |
| `broker::context` | `ctx_` | 8 | (date, ticker) | 60d |
| `broker::temporal` | `tfl_` | 30 | (date, ticker) | 5/20/60 |
| `broker::localfund` | `localfund_` | 24 | (date, ticker) | 5/20/60 |
| `broker::bandar` | `bandar_` | 11 | (date, ticker) | 5/20/60 |
| `broker::key_mg` | `mg_` | 6 | (date, ticker) | 0 |
| `broker::key_xl` | `xl_` | 4 | (date, ticker) | 0 |
| `broker::cross_sectional` | `sq_`, `xc_`, `yp_`, `pd_` | 24 | (date, ticker) | 0 |
| `broker::breadth` | `buyer_`, `seller_`, `broker_` | 7 | (date, ticker) | 30 |
| `ohlcv::momentum` | `close_` | 5 | (date, ticker) | 0 |
| `ohlcv::overnight` | `overnight_`, `gap_`, `gapdown_` | 30 | (date, ticker) | 5/20/60 |
| `ohlcv::derived` | `yf_` | 58 | (date, ticker) | 5/20/60 |
| `global::indices` | `ihsg_`, `usdidr_`, `vix_`, `nasdaq_`, `nikkei_` | 15 | (date,) | 200d |
| `cvd::features` | `cvd_` | 6 | (date, ticker) | 20 |
| | **TOTAL** | **~249** | | |

Note: VWAP is explicitly out of scope (1 column, requires tick data). Fill with `0.0` or `f64::NAN` per model expectations.

### 3.3 Model Inference (predict)

```
predict(variant):
  1. Load model from model/BSJP/{variant}/model_lightgbm_*.txt
  2. SELECT * FROM features_store WHERE date = MAX(date)
  3. For each ticker: model.predict_proba(feature_vector)
  4. Sort descending by pred_proba
  5. Take top-3 → apply rank weights → log to picks_log
```

**LightGBM inference options (ordered by preference):**

1. **`lightgbm` crate (C FFI)** — Call lib_lightgbm.so directly via FFI bindings. Fastest, zero-overhead, but requires runtime lib_lightgbm.so on deployment target.
2. **Treelite (treelite-rs)** — Convert model to `.so` shared library at build time. Portable, optimized inference via TVM. Best for VPS deploy.
3. **Native decision-tree walk** — Parse the LightGBM `model.txt` at startup, implement tree traversal in pure Rust. No FFI, fully portable, but requires careful implementation and benchmarking.

For GA, evaluate (1) first (fastest to implement), fall back to (3) if FFI causes deploy friction. Treelite (2) is ideal end-state but has build complexity.

### 3.4 DuckDB Schema

Identical to the Python `db.py` schema:

```sql
CREATE TABLE IF NOT EXISTS features_store (
    date        DATE NOT NULL,
    ticker      VARCHAR NOT NULL,
    entry_price DOUBLE,
    "close_ret_last1h" DOUBLE,
    -- ... 247 more feature columns ...
    inserted_at TIMESTAMP DEFAULT now(),
    PRIMARY KEY (date, ticker)
);

CREATE TABLE IF NOT EXISTS picks_log (
    date             DATE NOT NULL,
    variant          VARCHAR NOT NULL,
    rank             INTEGER NOT NULL,
    ticker           VARCHAR NOT NULL,
    pred_proba       DOUBLE,
    entry_price      DOUBLE,
    exit_price       DOUBLE,
    overnight_return DOUBLE,
    hit_tp           BOOLEAN,
    logged_at        TIMESTAMP DEFAULT now(),
    PRIMARY KEY (date, variant, rank)
);
```

Upsert strategy: `DELETE FROM features_store WHERE date = $target_date` → `INSERT`. Avoids dynamic column count from `ON CONFLICT ... DO UPDATE`.

---

## 4. Performance Budget

### 4.1 Per-Operation Targets

| Operation | Target | Strategy |
|---|---|---|
| Load 60-day L0 window | < 2 seconds | Polars lazy, projection pushdown |
| Compute broker features (L1) | < 30 seconds | Polars lazy, group_by + window |
| Compute OHLCV features | < 5 seconds | Simple rolling aggregations |
| Compute cross-sectional | < 5 seconds | per-date rank/percentile |
| Merge all modules | < 3 seconds | Lazy chain, single .collect() |
| Upsert to DuckDB | < 2 seconds | Bulk INSERT |
| LightGBM predict (all tickers) | < 1 second | Native tree walk or FFI |
| **TOTAL (fetch + predict)** | **< 60 seconds** | Well within 5-min budget |

### 4.2 Memory Budget

| Component | Memory |
|---|---|
| L0 60-day DataFrame (all sources) | ~80 MB on disk, ~200 MB in memory |
| L1 feature intermediates | ~100 MB peak |
| Final (date, ticker) DataFrame (157 tickers × 1 date) | < 5 MB |
| DuckDB in-memory | ~50 MB |
| LightGBM model | < 5 MB |
| **Peak RSS** | **< 400 MB** |

Headroom: 10× below 4 GB limit.

---

## 5. Project Structure

```
inferences/bsjp/rust/
├── PRD.md                    ← this file
├── Cargo.toml
├── Cargo.lock
├── rust-toolchain.toml       ← pin 1.95.0
├── src/
│   ├── main.rs               ← CLI entrypoint (clap)
│   ├── lib.rs                ← public API
│   ├── config.rs             ← paths, WIB timezone, constants
│   ├── error.rs              ← thiserror enum
│   ├── broker/
│   │   ├── mod.rs
│   │   ├── flow.rs
│   │   ├── context.rs
│   │   ├── temporal.rs
│   │   ├── buckets.rs         ← localfund, bandar
│   │   ├── key_brokers.rs     ← MG, XC, SQ, YP, PD, XL
│   │   ├── cross_sectional.rs ← sq_, xc_, yp_, pd_
│   │   └── breadth.rs
│   ├── ohlcv/
│   │   ├── mod.rs
│   │   ├── momentum.rs
│   │   ├── overnight.rs
│   │   └── derived.rs
│   ├── global.rs
│   ├── cvd.rs
│   ├── duckdb.rs              ← schema, upsert, query, bootstrap
│   ├── model.rs               ← LightGBM tree walk or FFI
│   └── calibrate.rs           ← comparison vs training pipeline
└── tests/
    ├── integration.rs         ← end-to-end on sample data
    └── calibration.rs         ← MAE comparison tests
```

---

## 6. Implementation Phases

| Phase | Artifact | Duration | Dependencies | Exit Criteria |
|---|---|---|---|---|
| **P0: Skeleton** | CLI + DuckDB schema + L0 parquet read | Day 1-2 | — | `bsjp fetch` reads L0, `bsjp bootstrap` backfills DuckDB |
| **P1: Broker features** | flow, ctx, temporal, buckets, key brokers | Day 3-5 | P0 | 103 broker columns computed |
| **P2: OHLCV + Global** | momentum, overnight, derived, global indices | Day 5-7 | P1 | Full feature vector available |
| **P3: Cross-sectional + CVD** | sq_/xc_/yp_/pd_, cvd | Day 7-8 | P2 | 249 features complete |
| **P4: LightGBM inference** | model loading, predict, picks_log | Day 8-10 | P0 | `bsjp predict --variant v15` output matches Python |
| **P5: Calibration** | 5-date comparison, MAE report | Day 10-12 | P3 | All 249 features MAE < 1e-10 |
| **P6: Polish** | error handling, tracing, --help docs, release build CI | Day 12-14 | P5 | Single binary ready for VPS deploy |

### 6.1 Calibration Protocol

```bash
for date in 2026-04-17 2026-04-18 2026-04-21 2026-04-22 2026-04-23; do
  # Python (training pipeline, ground truth)
  python fetch_lightweight.py --date $date --force

  # Rust (this binary)
  bsjp fetch --date $date

  # Compare
  bsjp calibrate --date $date
done
```

For each date, `calibrate` exports both versions from DuckDB, joins on `(date, ticker)`, and computes per-column MAE. Any column with MAE > 1e-10 is flagged and investigated.

---

## 7. Deployment

### 7.1 Build

```bash
cargo build --release
strip target/release/bsjp         # ~50 MB → ~15 MB
```

### 7.2 VPS Checklist

```bash
# 1. scp binary
scp target/release/bsjp vps:/opt/bsjp/

# 2. One-time bootstrap
./bsjp bootstrap --l2-parquet data/Level_2_Datamart/training_datamart_bsjp_overnight.parquet

# 3. Cron (WIB)
#   17:30 Mon-Fri
#   ./bsjp fetch && ./bsjp predict --variant v15 --log
```

### 7.3 Monitoring

- Exit code != 0 triggers alert
- `./bsjp check` returns `MAX(date)` + days since last update
- `./bsjp predict` logs top-3 to both DuckDB and stdout

---

## 8. Risk Register

| Risk | Prob | Impact | Mitigation |
|---|---|---|---|
| Polars lazy query plan diverges from pandas semantics | Medium | High | Calibration phase (P5) gates GA |
| LightGBM C FFI unavailable / linking issues on VPS | Medium | Medium | Fallback to native tree walk (P4) |
| Cross-sectional features (sq_, xc_) require full universe — polars window vs pandas group_by | Medium | Medium | Extra calibration focus on cross-sectional columns |
| VPS missing system libraries (glibc version) | Low | High | Statically link musl target |
| L0 parquet schema drift (new columns, renamed) | Low | Medium | Schema validation on startup, warn but don't crash on extra columns |
| Bit-identical impossible due to fp rounding differences | Low | Low | Accept MAE < 1e-6 if root cause is fp rounding; document |

---

## 9. Open Questions

| # | Question | Owner | Due |
|---|---|---|---|
| Q1 | Which LightGBM inference strategy? (FFI vs treelite vs native) | Engineering | Before P4 |
| Q2 | Static musl linking or dynamic glibc? (VPS distro) | DevOps | Before P6 |
| Q3 | Should `bootstrap` remain a separate subcommand or merge into `fetch --backfill`? | Engineering | P0 |
| Q4 | VWAP: fill NaN, 0.0, or skip the 6 columns entirely? | Quant Research | Before P3 |
| Q5 | Should calibration use `assert_frame_equal` from Python or reimplement in Rust? | Engineering | P5 |

---

## 10. Revision History

| Date | Version | Author | Changes |
|---|---|---|---|
| 2026-04-28 | 0.1 | Engineering | Initial draft; skeleton compiled, L0 read verified |
