# MMMACHINE / idx (BPJS & BSJP Screener)

Sistem screener fully-automated untuk saham IDX non-blue chip. Sistem mengeksekusi strategi trading berbasis aktivitas *broker summary* yang dikuantifikasi menjadi probabilitas empiris. Tidak ada intervensi manual atau keputusan berbasis intuisi—murni berdasarkan *data-driven model*.

## Arsitektur Sistem (3 Lapisan Data + Model)

Sistem dibangun dalam 3 lapisan pipeline data yang berjalan setiap hari, ditambah layer model:

1. **Level 0 (Raw Data Fetching):**
   - Menarik data *broker summary* (IPOT WebSocket), OHLCV intraday & harian (yfinance), dan global indices (^IXIC, ^N225, ^VIX, USD/IDR, IHSG).
   - Data master emiten (IDX / D1 API) dan klasifikasi broker (Foreign, Local Fund, Retail).
   - Dijalankan via shell script setiap sore (17:35 - 18:20 WIB).
   - **Grain:** bervariasi per sumber.
   - **Path:** `data/Level_0_Raw/`

2. **Level 1 (Feature Engineering):**
   - Membangun feature mart strategy-agnostic dari raw broker + market data.
   - Fitur mencakup: rasio mikrostruktur (churn ratio, net flow ratio, velocity), rolling MA/Std/Z-score (window 5/20/60), aktivitas broker, price context, dan market features dari yfinance.
   - Semua fitur menggunakan `shift(1)` untuk menghindari *look-ahead bias*.
   - Dihasilkan oleh `pipeline/feature/generate_broksum_datamart.py`.
   - **Grain:** `1 row = 1 date + 1 broker + 1 ticker`
   - **Path:** `data/Level_1_Features/broksum_datamart.parquet`

3. **Level 2 (Training Datamart & Labeling):**
   - Mengagregasi fitur Level 1 menjadi **1 baris per ticker per hari** (1-Row Rule).
   - Melakukan agregasi perilaku broker berdasarkan kategori (Foreign, Local Fund, Retail), sinyal broker spesifik (MG/bandar), breadth (broker count, buyer/seller ratio).
   - Menggabungkan konteks *market regime* (HMM dari IHSG), global indices, dan label target return.
   - Dihasilkan per edge oleh `edges/<edge>/scripts/generate_datamart.py`.
   - **Grain:** `1 row = 1 date + 1 ticker`
   - **Path:** `data/Level_2_Datamart/`

4. **Model (LightGBM Binary Classifier):**
   - **LightGBM** binary classifier dengan walk-forward validation (OOT split).
   - Input: canonical feature vector dari Level 2 (`model.feature_name()`; v15 memakai 249 fitur).
   - Output per training: model file, metrics.json (AUC OOT, cum_return, max_dd), feature importance, valid predictions.
   - Policy: Top-3 per hari. Policy/sizing harus dibaca dari `metrics.json` masing-masing model; jangan asumsi semua model memakai 60-30-10.
   - **Path:** `model/BPJS/` dan `model/BSJP/`

## Data Handling: Training vs Production Inference

Ada dua jalur data yang sengaja dipisah. Ini penting untuk mencegah agent berikutnya menjalankan ulang pipeline mahal saat hanya butuh sinyal harian.

### 1. Parquet = Historical Archive & Training Source

Parquet tetap menjadi sumber untuk research, audit, dan retraining.

| Layer | File | Peran |
|---|---|---|
| L0 raw | `data/Level_0_Raw/*.parquet` | Arsip historis broker summary, OHLCV, master emiten/broker, global indices |
| L1 feature mart | `data/Level_1_Features/broksum_datamart.parquet` | Feature engineering strategy-agnostic untuk training/batch analysis |
| L1 VWAP | `data/Level_1_Features/vwap_features.parquet` | Fitur VWAP tambahan untuk L2 BSJP |
| L2 training | `data/Level_2_Datamart/training_datamart_bsjp_*.parquet` | Datamart training/backtest BSJP, grain `(date,ticker)`; objective must be explicit (`overnight` exit 09 vs `close10` exit 10) |
| Model artifacts | `model/BSJP/bsjp_v*/` | `metrics.json`, model LightGBM, predictions, portfolio, plots |

Catatan operasional:
- Full/batch rebuild L1/L2 boleh dilakukan untuk training, audit, dan retrain.
- Jangan pakai single-file L1/L2 parquet rewrite sebagai default production inference path.
- `generate_broksum_datamart.py` sudah punya date filter dan warning, tetapi incremental upsert single-file masih membaca existing output dan rewrite file besar.
- Kalau perlu low-latency production, pindahkan state harian ke DuckDB/partitioned storage, bukan memaksa single parquet besar.

### 2. DuckDB = Production Inference Feature Store

Production inference memakai DuckDB agar scoring harian tidak bergantung pada rewrite parquet besar.

| Store | Path | Peran |
|---|---|---|
| Inference DB | `inferences/bsjp/db/inference.duckdb` | Store feature vector siap-score dan picks log |
| Feature table | `features_store` | `date`, `ticker`, `entry_price`, dan canonical model features |
| Picks table | `picks_log` | Log rekomendasi per variant/date/rank |

Script penting:
- `inferences/bsjp/bootstrap_feature_store.py`: backfill feature vectors dari L2 parquet ke DuckDB. Dipakai saat bootstrap awal atau saat model feature list berubah.
- `inferences/bsjp/fetch.py`: fast-path vector extraction/upsert. Default-nya tidak rebuild L1/L2.
- `inferences/bsjp/run.py`: scoring dari DuckDB `features_store`.

Kontrak `fetch.py` terbaru:
- `--force` hanya re-upsert vector ke DuckDB, bukan rebuild L1/L2.
- Jika L1/L2 belum punya target date, script stop dengan pesan eksplisit.
- Slow rebuild hanya bisa dipanggil sengaja dengan `--rebuild-l1` atau `--rebuild-l2`.
- Ini dibuat supaya agent tidak sengaja menjalankan `generate_broksum_datamart.py` yang membaca jutaan row untuk kebutuhan inference.

Benchmark terakhir:
- Fast path `fetch.py --date 2026-04-23 --force`: 3.35 detik, extract/upsert 157 vectors, tidak menyentuh mtime L1/L2 parquet.
- Bootstrap 1 hari ke DuckDB: 2.88 detik untuk 157 vectors.
- Bootstrap full L2 ke DuckDB: 8.74 detik untuk 113,263 vectors, 730 trading days.
- Scoring `run.py --variant v15 --date 2026-04-23`: 2.58 detik dari DuckDB.

Status production gap:
- DuckDB vector store sudah ada dan tervalidasi.
- Yang masih perlu dibuat untuk production-grade end-to-end adalah daily builder yang membaca L0 terbaru + warmup terbatas, menghitung vector target date, lalu insert langsung ke DuckDB tanpa rewrite single-file L1/L2.
- Sampai daily builder itu ada, L1/L2 parquet tetap dipakai untuk training/backfill, bukan sebagai jalur utama sinyal harian.

## Struktur Direktori

```text
idx/
├── pipeline/           # Infrastruktur pipeline data
│   ├── fetch/          #   Fetcher: broksum, yfinance, emiten, broker, global indices
│   ├── feature/        #   Level 1 feature engineering
│   └── run/            #   Shell scripts untuk cron jobs
├── inferences/         # Production inference: DuckDB feature store, variant config, runner
│   └── bsjp/           #   features_store, picks_log, v15 fetch/run/bootstrap scripts
├── data/               # Data lake
│   ├── Level_0_Raw/    #   Raw parquet: broksum_bybroker, yfinance_1h/4h/daily, global_indices, master_emiten, master_broker
│   ├── Level_1_Features/ # Feature mart: broksum_datamart.parquet
│   └── Level_2_Datamart/ # Training datamart per edge + label
├── edges/              # Satu folder per strategi trading
│   ├── bpjs_opening_tp3/ # Beli Pagi Jual Siang (09:00-10:00, TP +3%, SL -3%) — PAUSE
│   │   ├── edge.md       #   Definisi strategi
│   │   ├── scripts/      #   generate_datamart.py + train_lightgbm.py
│   │   └── analysis/     #   Notebook/skrip evaluasi
│   └── bsjp_overnight_sl2/ # BSJP research edge; current target objective is close T → open 10:xx T+1
│       ├── edge.md        #   Definisi strategi (best: v7, AUC 0.602, return +103%)
│       ├── scripts/       #   generate_datamart.py + train_lightgbm.py
│       └── analysis/      #   Evaluasi hasil prediksi
├── model/              # Iterasi model hasil training
│   ├── BPJS/           #   30 iterasi (v1-v18), AUC OOT ~0.72 tapi return negatif
│   └── BSJP/           #   10 iterasi (v1-v7), best v7 AUC 0.602 return +103%
├── _DOC/               # Dokumentasi
│   ├── _PRD/           #   PRD: 0000_bpjs_screener, 0001_broker_enrichment, 0002_training_datamart
│   └── _TEST/          #   Test cases
├── _LOG/               # Log file eksekusi fetcher harian
├── _ARCH/              # Arsitf eksperimen/notebook lama
├── _BAK/               # Backup file konfigurasi
├── _MEMORY/            # Session memory / timestamped notes
├── .claude/            # Claude project config
├── requirements.txt    # Python dependencies
└── README.md
```

## Workflow Harian (Cron Pipeline)

Sistem punya dua workflow berbeda: batch training/research dan production inference.

### Batch Training / Research

```bash
# 1. Fetch master data (jarang berubah)
bash pipeline/run/run_fetch_emiten.sh
bash pipeline/run/run_fetch_master_broker.sh

# 2. Fetch market data (yfinance)
bash pipeline/run/run_fetch_yfinance_daily.sh
bash pipeline/run/run_fetch_yfinance.sh

# 3. Fetch broker activity (IPOT)
bash pipeline/run/run_fetch_broksum.sh

# 4. Build Level 1 features
bash pipeline/run/run_feature_l1.sh

# 5. Build Level 2 training datamart
cd edges/bsjp_overnight_sl2/scripts
python generate_datamart.py

# 6. Train/evaluate model
python train_lightgbm.py \
  --output-dir ../../../model/BSJP/bsjp_vNEXT \
  --feature-prune-top-n 0 \
  --tp-pct 0.01 \
  --sl-pct -0.02 \
  --oot-valid-days 100
```

### Production Inference

```bash
# One-time or after model feature-list changes:
python inferences/bsjp/bootstrap_feature_store.py --replace

# Daily scoring when L1/L2/vector source is already fresh:
python inferences/bsjp/fetch.py --date YYYY-MM-DD --force
python inferences/bsjp/run.py --variant v15 --date YYYY-MM-DD --log-picks
```

Penting:
- `inferences/bsjp/fetch.py` adalah fast vector-store updater, bukan full data builder.
- Jangan pakai `--rebuild-l1` atau `--rebuild-l2` di production cron kecuali memang sengaja menerima slow path.
- Dedicated daily vector builder harus menjadi production writer setelah diimplementasikan.

## Strategi

### BSJP (Beli Sore Jual Pagi) — **ACTIVE ✅**

| Item | Detail |
|---|---|
| Entry | Close ~15:30-15:45 WIB (candle 15:xx hari T) |
| Exit | Current target objective: open 10:xx WIB T+1 (`close10`). Some older production variants use open 09:xx (`overnight`). |
| Label | Close10: `(open@10:xx_T+1 - close@15:xx_T) / close@15:xx_T`; Overnight: `(open@09:xx_T+1 - close@15:xx_T) / close@15:xx_T` |
| TP | `overnight_return > 0.4%` (roundtrip cost) |
| SL | `overnight_return < -2%` (klasifikasi saja) |
| Model | LightGBM binary classifier |
| Policy | Top-3 per hari; sizing tergantung artifact model |
| Active inference | `inferences/bsjp/variants/v15.py` |
| Current v15 | AUC OOT `0.60198`, cum_return `9.74255`, max_dd `-15.25%`, 249 features |

Top features: closing momentum (`close_ret_last1h` dominan), overnight history (gap up rate 20-60d), daily OHLCV, global macro (VIX), gap down profile (overnight_p10_5d — baru di v7).

Catatan v15:
- `bsjp_v15` dilatih sebagai BSJP overnight: beli close/hour-15 hari T, jual open/hour-9 T+1.
- Metadata `northstar` di `metrics.json` sempat salah menulis `09:00 -> 10:00`; sudah dikoreksi agar sesuai label aktual.
- Repro test `bsjp_v15_1` dengan parameter identik menghasilkan metrics, predictions, dan portfolio daily yang sama persis dengan v15 (`pred_proba_max_abs_diff = 0.0`).

Catatan v18 / close10:
- Forensik Apr 29 menunjukkan OLD `training_datamart_bsjp_overnight.parquet` yang dipakai v18 sebenarnya berisi `label_name = bsjp_close10_sl2` dengan exit `open@10 T+1`; nama file lama misleading.
- NEW `training_datamart_bsjp_overnight.parquet` adalah true overnight exit `open@09 T+1`, sehingga label berbeda besar dan model collapse ke 27/28 trees.
- Rebuild close10 yang menyerupai OLD dibuat di `data/Level_2_Datamart/training_datamart_bsjp_close10_rebuild_v18like.parquet`.
- Model kandidat `model/BSJP/bsjp_v18_close10_rebuild/`: 249 features, 408 trees, OOT AUC 0.650. Raw k3/w34 policy gives mean daily net +1.30% with MaxDD -35.6%.
- Recommended paper-trade artifact: `model/BSJP/bsjp_v18_close10_rebuild_policy_w25/`, same model with k=3 and max weight 25%. OOT mean daily net +0.98%, MaxDD -28.2%; Monte Carlo 100d median 2.57x, P(loss)=1.78%, P(MaxDD≤-30%)=7.61%.

### BPJS (Beli Pagi Jual Siang) — **PAUSE ⏸️**

| Item | Detail |
|---|---|
| Entry | Open 09:00 WIB |
| Exit | Time-based: tepat 10:00 WIB |
| TP | `close@10:00 >= open@09:00 * 1.02` (+2%) |
| SL | `min_return < -3.5%` |
| Model | LightGBM binary classifier |
| Best AUC | ~0.723 (v16b), tapi return negatif |

**Masalah fundamental:** Window 1 jam terlalu sempit, break-even win rate 71% tapi hanya tercapai ~42%. Over-regularisasi (best_iteration selalu 4-10 dari 4000). Direkomendasikan untuk di-pause sampai regularisasi di-fix dan cross-sectional features ditambahkan.

## Eksekusi Model (Training)

```bash
# Training BSJP (ACTIVE)
cd edges/bsjp_overnight_sl2/scripts
python generate_datamart.py --strategy-mode bsjp
python train_lightgbm.py --output-dir ../../model/BSJP/bsjp_v8 --feature-prune-top-n 0 --tp-pct 0.01 --sl-pct -0.02 --oot-valid-days 100 --rank-weights "0.6,0.3,0.1"

# Training BPJS (PAUSED)
cd edges/bpjs_opening_tp3/scripts
python generate_datamart.py --strategy-mode bpjs
python train_lightgbm.py --output-dir ../../model/BPJS/bpjs_lgbm_v19
```

*Output per training:* `metrics.json`, `model_lightgbm_*.txt`, `feature_importance.csv`, `valid_predictions.parquet`.

## Pipeline Data (End-to-End)

```
Level 0 (Raw)                       Level 1 (Features)          Level 2 (Datamart)         Model
───────────                         ──────────────────          ───────────────────         ─────
broksum_bybroker.parquet  ───┐
yfinance_1h.parquet       ───┤
yfinance_4h.parquet       ───┤──> generate_broksum_datamart.py
yfinance_daily.parquet    ───┤    (normalize → enrich →        broksum_datamart.parquet
master_emiten.parquet     ───┤     base → context → temporal     (122 cols,
master_broker.parquet     ───┘     → activity → market merge     broker×ticker×date)
global_indices.parquet    ───┘     → prefix → finalize)
                                                              └──> generate_datamart.py
                                                                   (aggregate broker →
                                                                    breadth → HMM →
                                                                    label → finalize)
                                                                  training_datamart.parquet
                                                                   (234 cols, ticker×date)
                                                                                          └──> train_lightgbm.py
                                                                                               (walk-forward →
                                                                                                binary classifier)
                                                                                              model/BSJP/vNN/
```

## Data Lake

| Layer | Path | Grain | Isi |
|---|---|---|---|
| Level 0 | `data/Level_0_Raw/` | per sumber | broksum_bybroker, yfinance_1h/4h/daily, global_indices, master_emiten, master_broker |
| Level 1 | `data/Level_1_Features/` | (date, broker, ticker) | broksum_datamart (~122 kolom) |
| Level 2 | `data/Level_2_Datamart/` | (date, ticker) | training_datamart_*; BSJP v15 L2 punya 260 kolom |
| Inference | `inferences/bsjp/db/inference.duckdb` | (date, ticker) | production feature vectors + picks log |

## Modular Feature Loading (Proposed Architecture)

**Problem:** [`generate_datamart.py`](edges/bsjp_overnight_sl2/scripts/generate_datamart.py) is a monolithic ETL script. Adding a new feature requires: editing the script, adding merge logic, rebuilding the full L2 parquet (~3 minutes), with risk of breaking existing features.

**Solution:** Separate each feature group into individual parquet modules that are auto-discovered at training time.

### Module Directory

```
data/Level_1_Features/modules/               # NEW
├── closing_momentum_features.parquet        # grain: (date, ticker)
├── overnight_history_features.parquet
├── broker_aggregate_features.parquet
├── cvd_features.parquet
├── global_indices_features.parquet
├── stockbit_xl_features.parquet
└── vwap_features.parquet
```

### How It Works

[`train_lightgbm.py`](edges/bsjp_overnight_sl2/scripts/train_lightgbm.py) gets a new `--feature-modules-dir` argument:

```bash
python train_lightgbm.py \
  --output-dir model/BSJP/bsjp_vNEXT \
  --feature-modules-dir data/Level_1_Features/modules/ \
  --feature-prune-top-n 0
```

At startup, the script:
1. Globs `*_features.parquet` in the modules directory
2. Loads each parquet, verifies `grain = (date, ticker)`
3. LEFT JOINs each module onto the base training DataFrame
4. Feature columns are auto-discovered (all non-key columns)
5. Missing modules are silently skipped → **backward compatible**

### Zero Quality Tradeoff

Both paths (monolithic vs modular) use:
- Same L0/L1 source parquets
- Same feature engineering logic
- Same LEFT JOIN on `(date, ticker)`
- Same `float32` dtype casting
- Same `shift(1)` no-lookahead guarantees

The resulting DataFrame is **byte-identical**. Verify with `pd.testing.assert_frame_equal()`.

### Performance Targets

| Metric | Monolithic (Current) | Modular (Target) | Improvement |
|--------|---------------------|-------------------|-------------|
| Full L2 rebuild | ~3 min | **< 1 min** | ~3x faster |
| Add new feature group | edit + rebuild all | drop parquet file | instant |
| Incremental update | ~3 min | ~5-10s | ~20x |

### Agility in Feature Exploration

The key benefit: data scientist creates a new feature notebook → computes feature vector per `(date, ticker)` → exports to `modules/my_new_feature_features.parquet` → runs `train_lightgbm.py` — **no code changes to generate_datamart.py**. Model automatically discovers new features via `choose_feature_columns()`.

### Success Metrics

| Metric | Definition | Sanity Check |
|--------|------------|--------------|
| **Speed** | Full rebuild < 60 seconds | Target tercapai ✅ |
| **Quality** | Random check matches old parquet exactly | `assert_frame_equal` on 10,000 rows |
| **Agility** | Add feature without touching generate_datamart.py | New `*_features.parquet` in modules dir |
| **Backward compat** | No `--feature-modules-dir` = original behavior | v18 retrain with identical args |

### Implementation Phases

1. ✅ **Phase 1 (Extract):** [`generate_datamart.py`](edges/bsjp_overnight_sl2/scripts/generate_datamart.py) — Added `--modules-dir` argument; writes each feature group to `modules/{name}_features.parquet` after build. Guarded by `modules_enabled` flag. 7 modules produced (245 MB total).
2. ✅ **Phase 2 (Load):** [`train_lightgbm.py`](edges/bsjp_overnight_sl2/scripts/train_lightgbm.py) — Added `--feature-modules-dir` argument + `load_features_from_modules()` function. Scans `*_features.parquet`, auto-detects grain `(date, ticker)` or `(date,)`, LEFT JOINs onto core columns. Verified: identical feature set (252/252), same row count.
3. **Phase 3 (Refactor):** Optionally make generate_datamart.py read modules instead of inline building.
4. **Phase 4 (Sanity):** Full rebuild + random check + document.

> **🏆 Recommendation:** Always use `--feature-modules-dir data/Level_1_Features/modules` when running `train_lightgbm.py`. Loading time drops from ~3 min to ~5-10s, feature set is verified identical (252/252), and adding new features is instant — just drop a `*_features.parquet` into the modules directory. The monolithic path (no flag) remains as backward-compatible fallback only.

### Data Flow Diagram (Updated)

```
Level 0 (Raw)                       Level 1 (Features)          Level 2 (Datamart / Modules)   Model
───────────                         ──────────────────          ────────────────────────────    ─────
broksum_bybroker.parquet  ───┐
yfinance_1h.parquet       ───┤
yfinance_4h.parquet       ───┤──> generate_broksum_datamart.py
yfinance_daily.parquet    ───┤    (normalize → enrich →        broksum_datamart.parquet
master_emiten.parquet     ───┤     base → context → temporal     (122 cols,
master_broker.parquet     ───┘     → activity → market merge     broker×ticker×date)
global_indices.parquet    ───┘     → prefix → finalize)
                                                               └──> generate_datamart.py          train_lightgbm.py
                                                                   (label → closing_momentum →     ├── load L2 (label + base)
                                                                    overnight_history → stockbit →   ├── scan modules/*_features.parquet
                                                                    broker_aggregate → CVD →         ├── LEFT JOIN each module
                                                                    global_indices → VWAP →           ├── choose_feature_columns()
                                                                    finalize)                         └── train walk-forward
                                                                                                    model/BSJP/vNN/
                                                                  OR
                                                                  (modular path):
                                                                  ├── label.parquet                  # (NEW) modular features
                                                                  ├── closing_momentum_features.parquet
                                                                  ├── overnight_history_features.parquet
                                                                  ├── broker_aggregate_features.parquet
                                                                  ├── cvd_features.parquet
                                                                  ├── global_indices_features.parquet
                                                                  ├── stockbit_xl_features.parquet
                                                                  └── vwap_features.parquet
```

## Pengembangan Edge Baru

Untuk mengembangkan strategi baru, buat folder `edges/<nama_edge>/` yang berisi:
- `edge.md` — Definisi strategi, parameter entry/exit, no-lookahead rules, data paths, feature families, model iterations.
- `scripts/generate_datamart.py` — Pipeline spesifik edge: agregasi L1 → L2 + labeling.
- `scripts/train_lightgbm.py` — Training + walk-forward validation.
- `analysis/` — Notebook/skrip untuk evaluasi hasil prediksi.

Data *Level 0* dan *Level 1* dipakai bersama (shared layer) untuk efisiensi dan mencegah duplikasi komputasi.

## Referensi Utama

- Spesifikasi Sistem: `_DOC/_PRD/0000_bpjs_screener.md`
- Layer 1 Enrichment: `_DOC/_PRD/0001_broker_enrichment.md`
- Layer 2 Training Data: `_DOC/_PRD/0002_training_datamart.md`
- Fetch Pipeline: `data/0000_FETCH_PROMPT.md`
- Data Warmup Readiness: `data/0001_DATA_WARMUP_READINESS.md`
- Model Training Config: `model/BPJS/0000_LIGHTGBM_TRAIN_PROMPT.md`
- Session Memory (latest): `_MEMORY/20260427053448.md`
