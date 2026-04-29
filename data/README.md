# Data Lake — MMMACHINE / idx

> **⚠️ Supersedes:** [`0000_FETCH_PROMPT.md`](0000_FETCH_PROMPT.md) dan [`0001_DATA_WARMUP_READINESS.md`](0001_DATA_WARMUP_READINESS.md) — kedua file tersebut **deprecated** dan tidak akan diupdate. Semua dokumentasi data terpusat di sini.

## Arsitektur Data (3 Lapisan)

```
Level 0 (Raw)        Level 1 (Features)          Level 2 (Training / Modules)
────────────────     ──────────────────          ─────────────────────────────────
broksum_bybroker     broksum_datamart            training_datamart_bsjp_overnight
yfinance_1h/4h/daily  └── vwap_features          training_datamart_bsjp_close10
global_indices        └── modules/               label_bpjs_intraday
master_emiten             ├── closing_momentum
master_broker             ├── overnight_history
                          ├── broker_aggregate
                          ├── cvd
                          ├── global_indices
                          ├── stockbit_xl
                          └── vwap
```

### Level 0 — Raw Data (`Level_0_Raw/`)

| File | Source | Grain | Contents |
|------|--------|-------|----------|
| [`broksum_bybroker.parquet`](Level_0_Raw/broksum_bybroker.parquet) | IPOT WebSocket | `(date, broker, ticker)` | Broker summary: cumulative volume, net flow by broker per ticker |
| [`yfinance_1h.parquet`](Level_0_Raw/yfinance_1h.parquet) | Yahoo Finance | `(datetime, ticker)` | OHLCV 1-hour bars |
| [`yfinance_4h.parquet`](Level_0_Raw/yfinance_4h.parquet) | Yahoo Finance | `(datetime, ticker)` | OHLCV 4-hour bars |
| [`yfinance_daily.parquet`](Level_0_Raw/yfinance_daily.parquet) | Yahoo Finance | `(date, ticker)` | OHLCV daily bars |
| [`global_indices.parquet`](Level_0_Raw/global_indices.parquet) | Yahoo Finance | `(date,)` | ^IXIC, ^N225, ^VIX, USD/IDR, IHSG — market-level only |
| [`master_emiten.parquet`](Level_0_Raw/master_emiten.parquet) | IDX/D1 API | `(ticker,)` | Master data emiten IDX |
| [`master_broker.parquet`](Level_0_Raw/master_broker.parquet) | IDX API | `(broker,)` | Master broker IDX + klasifikasi (Foreign/Local Fund/Retail) |
| [`ipot_ohlcv_1h.parquet`](Level_0_Raw/ipot_ohlcv_1h.parquet) | IPOT | `(datetime, ticker)` | OHLCV 1-hour bars alternatif |

**Fetch pipeline:** Dijalankan via shell script setiap sore (17:35-18:20 WIB). Lihat [`pipeline/run/`](../pipeline/run/) untuk detail cron.

### Level 1 — Feature Engineering (`Level_1_Features/`)

| File | Grain | Columns | Description |
|------|-------|---------|-------------|
| [`broksum_datamart.parquet`](Level_1_Features/broksum_datamart.parquet) | `(date, broker, ticker)` | ~122 | Strategy-agnostic feature mart. Rolling MA/Std/Z-score (5/20/60), broker activity, price context, market features. All features use `shift(1)` for no-lookahead. |
| [`vwap_features.parquet`](Level_1_Features/vwap_features.parquet) | `(date, ticker)` | ~8 | VWAP-derived features (VWAP, volume-weighted metrics) |
| [`modules/`](Level_1_Features/modules/) | varies | varies | **Modular feature parquets** — each feature family as a separate file (see below) |

#### Modular Feature Modules (`modules/`)

Setiap file di `modules/` adalah satu feature family yang bisa di-load independen oleh [`train_lightgbm.py`](../edges/bsjp_overnight_sl2/scripts/train_lightgbm.py) via `--feature-modules-dir`.

| Module | Size | Feats | Grain | Description |
|--------|------|-------|-------|-------------|
| [`closing_momentum`](Level_1_Features/modules/closing_momentum_features.parquet) | 879 KB | 3 | `(date, ticker)` | Momentum menjelang close |
| [`overnight_history`](Level_1_Features/modules/overnight_history_features.parquet) | 4.5 MB | 30 | `(date, ticker)` | Overnight return history |
| [`broker_aggregate`](Level_1_Features/modules/broker_aggregate_features.parquet) | 213 MB | 188 | `(date, ticker)` | Broker behavior aggregates |
| [`cvd`](Level_1_Features/modules/cvd_features.parquet) | 7.6 MB | 6 | `(date, ticker)` | CVD / delta features |
| [`global_indices`](Level_1_Features/modules/global_indices_features.parquet) | 117 KB | 15 | `(date,)` | Global index returns (market-level) |
| [`stockbit_xl`](Level_1_Features/modules/stockbit_xl_features.parquet) | 778 KB | 4 | `(date, ticker)` | Stockbit XL signals |
| [`vwap`](Level_1_Features/modules/vwap_features.parquet) | 19 MB | 6 | `(date, ticker)` | VWAP features |
| **Total** | **245 MB** | **252** | | |

**Menambah module baru:** Copy template [`_TEMPLATE_generate_feature_module.py`](../edges/bsjp_overnight_sl2/scripts/_TEMPLATE_generate_feature_module.py), rename, implement `build_features()`, jalankan. Output langsung terdeteksi oleh training tanpa edit kode.

> ⚠️ **Column name conflicts:** Jika dua module punya nama kolom sama, `load_features_from_modules()` akan warning dan pandas akan menambah suffix `_x`/`_y`. Selalu pakai `--dry-run` + `check_column_conflict()` sebelum menulis module baru.

### Level 2 — Training Datamart (`Level_2_Datamart/`)

| File | Strategy | Grain | Description |
|------|----------|-------|-------------|
| [`training_datamart_bsjp_overnight.parquet`](Level_2_Datamart/training_datamart_bsjp_overnight.parquet) | BSJP (Beli Sore Jual Pagi) | `(date, ticker)` | Datamart utama untuk strategi overnight. Entry ~16:00, Exit 09:00 T+1, SL -2%. ~109K rows, 260+ columns. |
| [`training_datamart_bsjp_close10.parquet`](Level_2_Datamart/training_datamart_bsjp_close10.parquet) | BSJP Close10 (Alternate) | `(date, ticker)` | Datamart alternatif dengan exit 10:00. |
| [`label_bpjs_intraday.parquet`](Level_2_Datamart/label_bpjs_intraday.parquet) | BPJS (Beli Pagi Jual Siang) | `(date, ticker)` | Label untuk strategi intraday BPJS. |

**Cara training (modular — recommended):**
```bash
python edges/bsjp_overnight_sl2/scripts/train_lightgbm.py \
    --training-path data/Level_2_Datamart/training_datamart_bsjp_overnight.parquet \
    --feature-modules-dir data/Level_1_Features/modules \
    ...
```
Ini hanya membaca 8 core columns dari L2 (date, ticker, entry_price, exit_price, label, dll) lalu LEFT JOIN semua module. Loading ~5-10 detik vs ~3 menit monolithic.

**Cara training (monolithic — fallback):**
```bash
python edges/bsjp_overnight_sl2/scripts/train_lightgbm.py \
    --training-path data/Level_2_Datamart/training_datamart_bsjp_overnight.parquet \
    ...
```
Membaca full 260+ columns dari L2 parquet (~700 MB). Backward compatible.

---

## Pipeline Workflow

### End-to-End Rebuild

```
1. Fetch L0 raw data     → pipeline/fetch/*.py         → Level_0_Raw/*.parquet
2. Build L1 feature mart  → pipeline/feature/*.py       → Level_1_Features/broksum_datamart.parquet
3. Build L2 training      → edges/<edge>/scripts/generate_datamart.py → Level_2_Datamart/*.parquet
   (also writes modules/  → same script, --modules-dir flag)
4. Train model            → edges/<edge>/scripts/train_lightgbm.py   → model/<Edge>/<run>/
```

### Daily Scoring (Production)

```
L0 latest → fetch.py (fast path, ~3s) → DuckDB features_store → run.py (scoring, ~2.5s)
```

See [`inferences/bsjp/`](../inferences/bsjp/) for production inference scripts.

---

## Integrity & Audit Checklists

Checklist berikut menggabungkan ekspektasi dari `0000_FETCH_PROMPT.md` dan `0001_DATA_WARMUP_READINESS.md`. Berlaku untuk semua operasi yang memodifikasi data layer.

### A. File & Schema

| # | Check | Method |
|---|-------|--------|
| 1 | File exists & size > 0 | `os.path.getsize()` |
| 2 | Max date recency | `df["date"].max()` vs expected |
| 3 | Required columns exist | Schema contract per layer |
| 4 | Critical dtype validation | `df.dtypes` match expected |
| 5 | No schema drift | Compare against reference dtypes |

### B. Key Uniqueness

| Layer | Unique Key | Tolerance | Action |
|-------|-----------|-----------|--------|
| L0 raw (broksum) | `(date, broker, ticker)` | 0 dupes | Hard fail |
| L1 feature mart | `(date, broker, ticker)` | 0 dupes | Hard fail |
| L2 training | `(date, ticker)` | 0 dupes | Hard fail |

### C. Row Reconciliation

| Step | Expected | Actual | Delta |
|------|----------|--------|-------|
| L0 → L1 | L0 after dedup | L1 rows | Reconciled |
| L1 → L2 | L1 aggregate (date,ticker) | L2 rows | Join coverage % |
| Label coverage | L2 ∩ label keys | % matched | ≥ 99% |

### D. Data Quality

| # | Check | Method | Hard Fail? |
|---|-------|--------|------------|
| 1 | Null ratio per column | `df.isna().mean()` | Report top-20 |
| 2 | All-null columns | Must be 0 | **Yes** |
| 3 | Invalid negatives | Where not allowed | Yes |
| 4 | Impossible ratios / infinities | `np.isinf()` | Yes |
| 5 | Extreme outliers | Robust z-score / IQR | Report |
| 6 | Global trading-day continuity | vs exchange calendar | Report gaps |
| 7 | Per-key continuity | Gap distribution | Max unexpected gap |
| 8 | No future-derived features | `shift(1)` verified | **Yes** |
| 9 | `feature_date < trade_date` | Strict alignment | **Yes** |

### E. Label Quality

| # | Check | BSJP Overnight | BPJS Intraday |
|---|-------|----------------|---------------|
| 1 | Positive rate | TP target +3% | TP target +3% |
| 2 | Class balance | ~40-50% TP | ~40-50% TP |
| 3 | Class drift by month | Per-month plot | Per-month plot |
| 4 | Return distribution | `close_return_next_day` | `return_to_1000` |

### F. Hard Fail Criteria (Stop on Any)

1. ✅ Duplicate keys > 0 on required unique constraints
2. ✅ All-null feature column in training table
3. ✅ Leakage column inside model feature candidates
4. ✅ Unexplained row loss beyond tolerance (default: > 2%)
5. ✅ Unexpected continuity hole above threshold
6. ✅ Missing required columns per schema contract

---

## Adding a New Feature Family

```bash
# 1. Copy template
cp edges/bsjp_overnight_sl2/scripts/_TEMPLATE_generate_feature_module.py \
   edges/bsjp_overnight_sl2/scripts/generate_myfeature_features.py

# 2. Edit: MODULE_NAME, GRAIN, load_sources(), build_features()

# 3. Dry-run (check column conflicts, NaN ratio)
python edges/bsjp_overnight_sl2/scripts/generate_myfeature_features.py --dry-run

# 4. Generate module
python edges/bsjp_overnight_sl2/scripts/generate_myfeature_features.py

# 5. Output → data/Level_1_Features/modules/myfeature_features.parquet
#    Training auto-detects on next run with --feature-modules-dir
```

---

## Referensi

- [README.md (root)](../README.md) — Dokumentasi utama proyek
- [edge.md (BSJP)](../edges/bsjp_overnight_sl2/edge.md) — Definisi strategi BSJP
- [edge.md (BPJS)](../edges/bpjs_opening_tp3/edge.md) — Definisi strategi BPJS
- [Session Memory (latest)](../_MEMORY/20260427053448.md) — Catatan sesi terbaru
- [PRD: Training Datamart](../_DOC/_PRD/0002_training_datamart.md) — Spesifikasi L2
