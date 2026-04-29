# PRD: Lightweight Inference Feature Pipeline

| Metadata | |
|---|---|
| **Status** | Draft |
| **Author** | Engineering |
| **Created** | 2026-04-28 |
| **Strategy** | BSJP Overnight |
| **Model** | v15 (249 features) |

---

## 1. Objective

Decouple production inference pipeline from the L1/L2 training pipeline.
Eliminate 15-hour rebuilds caused by freshness checks triggering full
parquet regeneration. Enable fully autonomous inference on a VPS without
any training infrastructure.

## 2. Background

### 2.1 Current Architecture

```
L0 Raw Data ──→ L1 Features ──→ L2 Datamart ──→ Inference (DuckDB)
                  (122 cols)      (249 cols)
```

The inference pipeline (`fetch.py`) checks whether L1 and L2 parquet files
contain data for today's date. If missing, it triggers a full rebuild:
- L1: `generate_broksum_datamart.py` scans all historical L0 (millions of rows)
- L2: `generate_datamart.py` rebuilds the training datamart from scratch

### 2.2 Incident: 2026-04-28

At 15:00 WIB, inference ran while broker data for the current day was
not yet available. The freshness check failed, triggering a full L1+L2
rebuild that took approximately 15 hours. During this time, no inference
output was produced. The DuckDB inference store remains stale at
2026-04-23 (3 trading days behind).

### 2.3 Root Cause

The inference pipeline conflates two concerns:
1. **Feature computation** — producing 249 feature columns for a target date
2. **Pipeline maintenance** — rebuilding L1/L2 parquet files

These should be separate.

## 3. Requirements

### 3.1 Functional

| ID | Requirement | Priority |
|---|---|---|
| F1 | Inference must produce top-3 picks within 15 minutes of invocation | P0 |
| F2 | Inference must run from L0 raw data only; zero dependency on L1/L2 | P0 |
| F3 | Must produce bit-identical features to the training pipeline (after calibration) | P1 |
| F4 | Must handle missing broker data gracefully: use T-2 data with a warning | P1 |
| F5 | Must support backfill: populate DuckDB with historical data from L2 | P1 |
| F6 | Must not modify any training pipeline files | P0 |

### 3.2 Non-Functional

| ID | Requirement | Target |
|---|---|---|
| N1 | Inference runtime (fetch + predict) | < 15 minutes |
| N2 | DuckDB historical data range | ≥ 2 years |
| N3 | Memory usage on VPS | < 4 GB RAM |
| N4 | Disk usage on VPS | < 2 GB (excluding model) |
| N5 | Portability | single Python file, no training libs |

### 3.3 Out of Scope

- VWAP feature computation (requires intraday tick data; will source from
  precomputed module or skip; only 6 of 249 features)
- HMM market regime (only used in BPJS, not BSJP)
- Training, walk-forward validation, or model selection
- L1/L2 pipeline optimization

## 4. Architecture

### 4.1 Data Flow (Target)

```
L0 Raw Data (on VPS, via rsync)
│
├── broksum_bybroker.parquet         ← 60-day window (~50 MB)
├── yfinance_1h.parquet              ← 60-day window (~20 MB)
├── yfinance_daily.parquet           ← 60-day window (~5 MB)
├── global_indices.parquet           ← 60-day window (<1 MB)
├── master_broker.parquet            ← static (<1 MB)
└── master_emiten.parquet            ← static (<1 MB)
         │
         ▼
┌──────────────────────────────────────────┐
│           fetch_lightweight.py           │
│                                          │
│  1. Load 60 days of L0 for rolling       │
│  2. Compute broker features (~103 cols)  │
│     - Base: flow_*, ctx_*                │
│     - Temporal: tfl_* MA/z/velocity      │
│     - Buckets: localfund, bandar, retail │
│     - Key brokers: MG, XC, SQ, YP, PD   │
│  3. Compute OHLCV features (~66 cols)    │
│     - Closing momentum (3 cols)          │
│     - Overnight history (30 cols)        │
│     - yf_* return/volume/range           │
│  4. Compute global context (15 cols)     │
│     - IHSG, USDIDR, VIX, Nasdaq, Nikkei │
│  5. Compute cross-sectional (24 cols)    │
│     - sq_, xc_, yp_, pd_ per date       │
│  6. Compute CVD + XL features (10 cols)  │
│  7. Merge all → feature vector           │
│  8. Upsert to DuckDB                     │
└──────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│               inference.duckdb           │
│  Table: features_store                   │
│  Grain: (date, ticker)                   │
│  Cols: date, ticker, entry_price, f1..f249      │
│  Size: ~1.5 GB (2 years × 700 tickers × 253 cols)│
└──────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│              run.py --variant v15        │
│  1. Load model from disk                 │
│  2. SELECT latest date FROM DuckDB       │
│  3. Predict proba for all tickers        │
│  4. Top-3 by pred_proba × rank_weights   │
│  5. Print picks + optional log           │
└──────────────────────────────────────────┘
```

### 4.2 Freshness Logic

```
fetch_lightweight.py --date YYYY-MM-DD:
  │
  ├─ Query DuckDB: SELECT MAX(date) FROM features_store
  │
  ├─ IF max_date >= target_date → SKIP (already up to date)
  │
  ├─ IF max_date < target_date:
  │     ├─ Read L0 + compute features for target_date
  │     ├─ Upsert to DuckDB
  │     └─ Print SUCCESS or WARNING
  │
  └─ IF compute fails (L0 missing):
        ├─ Print WARNING: "No broker data for {date}, using T-1"
        └─ Retry with target_date - 1
```

**Key rule:** Never trigger L1 or L2 rebuild. If data is missing, skip
the date and warn. The cron schedule should ensure L0 is fresh before
inference runs (e.g., run at 17:30 after broksum fetch completes).

### 4.3 VPS Deployment

```
Host: VPS (Linux, 4 GB RAM, 20 GB disk)

Cron (WIB):
  17:30 Mon-Fri → run_inference_bsjp.sh

run_inference_bsjp.sh:
  1. rsync L0 parquet files from dev (60-day window only)
  2. python fetch_lightweight.py         # compute + upsert
  3. python run.py --variant v15 --log-picks  # predict + log

Files on VPS (~500 MB total):
  inferences/bsjp/           ← Python modules (fetch, run, db, config, variants)
  model/BSJP/bsjp_v15/       ← model_lightgbm_opening_tp3.txt (single file)
```

## 5. Feature Transformation Specification

### 5.1 Broker Classification

Source: `master_broker.parquet`

| Category | Rule | Brokers (Example) |
|---|---|---|
| **Local Fund** | `category == "local fund"` | MG, etc. |
| **Foreign** | `category == "foreign fund"` | Various |
| **Retail** | `category == "retail"` | Various |
| **Bandar** | `localfund_% >= 60.0` | Subset of local fund |
| **Key: MG** | Focus broker (63% localfund) | "MG" |
| **Key: XC** | Retail (88%) | "XC" |
| **Key: SQ** | Retail (58%) | "SQ" |
| **Key: YP** | Retail (65%) | "YP" |
| **Key: PD** | Retail (74%) | "PD" |
| **Key: XL** | Stockbit broker | "XL" |

Retail bucket: `[XC, YP, PD, SQ]`

### 5.2 Required Transformations

#### 5.2.1 Broker Base Features (per broker×ticker×date)

```
gross_turnover       = gross_buy + gross_sell
net_flow_ratio       = total_net_buy / gross_turnover
buy_sell_ratio       = gross_buy / gross_sell
churn_ratio          = gross_turnover / |total_net_buy|
abs_net_buy          = |total_net_buy|
```

#### 5.2.2 Context Features (per broker×ticker×date)

```
broker_day_flow      = SUM(|net_buy|) per (date, broker)
ticker_day_flow      = SUM(|net_buy|) per (date, ticker)
market_day_flow      = SUM(|net_buy|) per (date)

broker_share         = |net_buy| / broker_day_flow
ticker_specificity   = |net_buy| / ticker_day_flow
market_share         = broker_day_flow / market_day_flow
net_buy_rank         = percentile_rank of net_buy within (date, ticker)
```

#### 5.2.3 Temporal Features (per broker×ticker, 60-day window)

After `shift(1)` (no lookahead):

| Feature | Window 5 | Window 20 | Window 60 |
|---|---|---|---|
| MA (mean) | ✓ | ✓ | ✓ |
| Std | — | ✓ | — |
| Z-score | — | ✓ | — |
| Velocity (value/mean) | ✓ | ✓ | ✓ |

Applied to: `total_net_buy`, `net_flow_ratio`, `churn_ratio`

Plus: `net_buy_sign_consistency_5` = 5-day mean of sign(shifted net_buy)

#### 5.2.4 L1→L2 Aggregation: Broker → Ticker

Group by `(date, ticker)` → for each numeric column: `sum` and `mean`.

Dimensions:
- ~90 numeric cols from L1 → 180 cols (90×_{sum}, 90×_{mean})
- ~115 numeric cols after additional features (broker count, buyer/seller ratio)

#### 5.2.5 Broker Bucket Aggregation

**Local Fund bucket** (all brokers where category == "local fund"):
- netbuy_sum, netbuy_mean, abs_netbuy_sum
- broker_count, buyer_count, seller_count
- HHI concentration, top1_share, top3_share
- participation_ratio, buyer_ratio, consensus_strength
- Rolling (ticker×date grain, with shift(1)):
  - netbuy_ma20, netbuy_std20, netbuy_z20
  - netbuy_3d_sum, netbuy_7d_sum
  - buy_freq_ma30, buy_freq_std30, buy_freq_z30

**Bandar bucket** (brokers with localfund_% >= 60):
Same schema as Local Fund.

#### 5.2.6 OHLCV Features

**Closing momentum** (from yfinance_1h, day T):
```
close_ret_last1h  = (close@15 - close@14) / close@14
close_vs_open_day = (close@15 - open@09)  / open@09
close_range_pct   = (day_high - day_low)  / open@09
```

**Overnight history** (from yfinance_1h/daily, with shift(1)):
```
overnight_ret[D] = (open@09[D] - close@15[D-1]) / close@15[D-1]
shifted = overnight_ret.shift(1)  # align to T-1
Then rolling on shifted:
  - MA 5/20/60
  - positive_rate 5/20/60 (fraction > 0)
  - worst 20/60 (rolling min)
  - p10 20/60 (rolling 10th percentile)
  - gapdown_freq 5/20/60 (fraction < -2%)
  - gapdown_severe_freq 5/20/60 (fraction < -5%)
  - gap_up2_freq 5/20/60 (fraction where high ≥ prev_close × 1.02)
  - gap_down2_freq 5/20/60 (fraction where low ≤ prev_close × 0.98)
  - gap_up2_down2_edge 5/20/60 (up2_freq - down2_freq)
  - gap_up2_followthrough_freq 5/20/60 (up2 days that close green)
```

#### 5.2.7 Global Context (from global_indices.parquet, all shift(1))

| Symbol | Features |
|---|---|
| `^JKSE` | prev_close, prev_return, MA(5), MA(20), MA(100), MA(200), close/MA5 ratio, close/MA20 ratio |
| `^IXIC` | prev_return |
| `^N225` | prev_return |
| `^VIX` | prev_close, 5d_avg |
| `IDR=X` | prev_close, prev_return, 5d_return |

#### 5.2.8 Cross-Sectional Features (per date, across all tickers)

These require the full universe of tickers for the target date:

- `sq_*`: squared values of selected features
- `xc_*`: cross-sectional rank / percentile of features within each date
- `yp_*`: yfinance feature percentiles (cross-sectional normalize)
- `pd_*`: pairwise ratios and derived statistics

### 5.3 Summary: 249 Features by Source

| Source Family | Count | Complexity | Notes |
|---|---|---|---|
| Broker: flow_ | 18 | Low | Base trading activity |
| Broker: ctx_ | 8 | Low | Market share, specificity |
| Broker: tfl_ | 30 | Medium | Rolling temporal |
| Broker: localfund_ | 24 | **High** | Fund bucket aggregation |
| Broker: bandar_ | 11 | **High** | Bandar bucket aggregation |
| Broker: mg_ | 6 | Low | Key broker MG |
| Broker: xc_, sq_, yp_, pd_ | 24 | Medium | Cross-sectional |
| Broker: xl_ | 4 | Low | Stockbit broker |
| Broker: buyer_, seller_ | 4 | Low | Broker counts |
| Broker: ctx_, broker_, total_ | 3 | Low | Market context |
| YFinance: yf_ | 58 | Medium | OHLCV derived |
| YFinance: close_ | 5 | Low | Closing momentum |
| Overnight/Gap: overnight_, gap_, gapdown_ | 30 | Medium | Overnight history |
| Global: ihsg_, usdidr_, vix_, nasdaq_, nikkei_ | 15 | Low | Macro context |
| CVD: cvd_ | 6 | Low | Volume accumulation |
| VWAP: vwap_ | 1 | **Skip** | From module or compute |
| Other: open_, vol_, last_ | 2 | Low | |

## 6. Implementation Plan

### 6.1 Phase 1: DuckDB Backfill

**Duration:** 15 minutes
**Files:** `bootstrap_feature_store.py` (existing)

Backfill historical data by extracting from the existing L2 parquet:
```bash
cd inferences/bsjp
python bootstrap_feature_store.py --replace
```

This is a one-time operation. The L2 parquet contains all 249 features
for 2023-03-07 through 2026-04-23.

### 6.2 Phase 2: Lightweight Fetch Script

**Duration:** 2-3 days
**Files:** `inferences/bsjp/fetch_lightweight.py` (new)

Implement the transformations specified in Section 5. The script must:
- Read 60 days of L0 data for rolling window computation
- Implement all broker, OHLCV, global, and cross-sectional transformations
- Support incremental operation (single date at a time)
- Skip gracefully if L0 data is missing for the target date

### 6.3 Phase 3: Calibration

**Duration:** 1 day

Run both pipelines for 5 sample dates and compare:

```bash
# Lightweight path
python fetch_lightweight.py --date 2026-04-23 --force

# Training path (existing)
python generate_broksum_datamart.py --date-from 2026-04-23 ...
python generate_datamart.py --date-from 2026-04-23 ...
python extract one row from L2

# Compare
python calibrate.py --lightweight-date 2026-04-23 --training-date 2026-04-23
```

Calibration script compares all 249 features per ticker and reports:
- MAE per feature across all tickers
- Max deviation per feature
- Features with MAE > 1e-10

If calibration fails, fix the lightweight transform and repeat.

### 6.4 Phase 4: Deploy to VPS

**Duration:** 0.5 days
**Files:** `pipeline/run/run_inference_bsjp.sh` (update)

Deployment script archives:
```
inferences/bsjp/
├── fetch_lightweight.py        ← NEW
├── run.py                      ← existing
├── config.py                   ← existing
├── db.py                       ← existing
├── variants/                   ← existing
model/BSJP/bsjp_v15/            ← model file only
pipeline/run/run_inference_bsjp.sh  ← updated
```

Initial VPS setup:
1. Install Python 3.12 + dependencies (lightgbm, pandas, pyarrow, duckdb)
2. Rsync inference module directory
3. Bootstrap DuckDB (extract from L2 parquet, 1 time)
4. Set up cron for 17:30 WIB
5. Set up daily rsync of L0 files from dev (60-day window)

### 6.5 Phase 5: Monitoring

- Inference log timestamps (fetch start, fetch end, predict)
- Feature freshness: `SELECT MAX(date) FROM features_store`
- Picks logged in DuckDB `picks_log` table
- Daily email/slack summary of top-3 picks

## 7. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Lightweight features differ from training pipeline | Medium | **High** | Calibration phase mandatory before deploy |
| Rolling windows diverge (60d vs full history) | High | Medium | Use 120d buffer; document delta |
| Cross-sectional ranking inconsistent | Medium | Medium | Compare distribution per date, not just mean |
| VWAP unavailable (needs tick data) | **High** | Low | Only 6/249 features; fill NaN with 0 |
| VPS memory insufficient for pandas | Medium | **High** | Use pyarrow + chunked reads; test on 4 GB VM |
| L0 rsync fails / stale | Low | Medium | Skip date with warning; retry next cron cycle |

## 8. Success Criteria

| Metric | Target |
|---|---|
| Inference runtime (fetch + predict) | < 15 minutes |
| Feature MAE (lightweight vs training) | < 1e-10 |
| Missing inference days (30-day window) | 0 |
| VPS disk usage (excluding model) | < 2 GB |
| Number of training files modified | 0 |

## 9. Appendix

### 9.1 Glossary

| Term | Definition |
|---|---|
| L0 | Raw data layer: broker summaries, yfinance OHLCV, global indices |
| L1 | Feature layer: broker-enriched data at (date, broker, ticker) grain |
| L2 | Training datamart: (date, ticker) grain with labels and all features |
| BSJP | Strategy: Beli Sore Jual Pagi (buy close, sell next-day open) |
| DuckDB | Embedded OLAP database for inference feature store |
| VWAP | Volume-Weighted Average Price |

### 9.2 Related Files

| Path | Purpose |
|---|---|
| `inferences/bsjp/fetch_lightweight.py` | Lightweight L0→DuckDB feature builder (NEW) |
| `inferences/bsjp/fetch.py` | Legacy inference fetcher (deprecated) |
| `inferences/bsjp/bootstrap_feature_store.py` | Historical backfill from L2 |
| `inferences/bsjp/run.py` | Model scoring + top-3 picks |
| `inferences/bsjp/db.py` | DuckDB schema + CRUD |
| `pipeline/run/run_inference_bsjp.sh` | Cron runner |
| `pipeline/feature/generate_broksum_datamart.py` | L1 pipeline (training only) |
| `edges/bsjp_overnight_sl2/scripts/generate_datamart.py` | L2 pipeline (training only) |

### 9.3 Revision History

| Date | Version | Author | Changes |
|---|---|---|---|
| 2026-04-28 | 1.0 | Engineering | Initial draft |
