# 2026-04-29 — Inferensi Go

## Target
`inferences/` mandiri — baca L0 langsung, komputasi fitur sendiri, zero Python.

## Architecture

```
L0 data (data/Level_0_Raw/)  →  Go compute  →  DuckDB  →  LightGBM native  →  top-3 picks
```

Dependency: nol Python. Bootstrap dari L2 parquet (sekali). Fetch harian dari L0 langsung.

## Status Per Feature Family

| Family | Source | Status | Catatan |
|--------|--------|--------|---------|
| Momentum (5) | yfinance_1h | ✅ byte-identical | hour-9/14/15 based, verified vs Python module |
| Overnight (30) | yfinance_1h | ⚠️ close | inner-join prev_close + position-based rolling. ma5/pos5 deket, gd5 perlu tweak |
| Broker (26) | broksum_bybroker | ✅ dari L0 | DuckDB SQL: rename + base ratios + context features → aggregate sum/mean → UPDATE features_store |
| Global (15) | global_indices | ⚠️ kode lama | per-date broadcast, belum diverifikasi |
| YfDaily (58) | yfinance_daily | ❌ | close_ma, volume_ma, range_ma — hitung tapi nilai salah |
| CVD (6) | L1 features | ❌ belum | |
| Stockbit (4) | broksum_bybroker | ❌ belum | |
| VWAP (~15) | vwap_features | ❌ belum | |

## Hasil Prediksi 2026-04-23 (vs production Python)

| # | Ticker | Go | Python | Status |
|---|--------|-----|--------|--------|
| 1 | MPMX | 0.5122 | 0.5122 | ✅ exact |
| 2 | IPCM | 0.5082 | 0.5020 | ⚠️ ε=0.006 |
| 3 | BELI | 0.4999 | 0.4999 | ✅ exact |

**Update 2026-04-29 session 2:** Overnight fixed:
- Prev_close sekarang pakai inner-join logic (skip days tanpa open9/close15), match Python merge
- Rolling pakai position-based (last W valid entries), match pandas rolling pada sparse series
- Hasil: gd5 AMAR match (0.20), ma5 AKRA deket (0.0206 vs 0.0227)
- Predict post-fix: GRIA/INDR/LPLI — overnight mendekati tapi belum byte-exact. Sisa epsilon dari edge case di gap feature computation

## File Kunci

- `inferences/bsjp/golang/cmd/bsjp/main.go` — CLI + merge helpers
- `inferences/bsjp/golang/internal/features/momentum.go` — closing momentum (byte-identical)
- `inferences/bsjp/golang/internal/features/overnight.go` — overnight history (perlu kalibrasi)
- `inferences/bsjp/golang/internal/features/broker_agg.go` — broker dari L0 via DuckDB SQL
- `inferences/bsjp/golang/internal/features/global.go` — global indices (lama)
- `inferences/bsjp/golang/internal/features/yfdaily.go` — yf_daily derived (belum fix)
- `inferences/bsjp/golang/internal/db/bootstrap.go` — bootstrap dari L2 parquet
- `inferences/bsjp/golang/internal/db/db.go` — DuckDB ops (FeatureRow map)
- `inferences/bsjp/golang/internal/model/lgb.go` — LightGBM native tree-walk (27 trees, sigmoid)

## Key Bugfixes

1. **Momentum**: close_ret_last1h pakai hour=15 vs hour=14 (bukan bar terakhir/kedua terakhir)
2. **Overnight prev_close**: inner-join — skip days tanpa open9. Bukan pakai adjacent calendar day.
3. **Overnight shift(1)**: ovs tersimpan di date next day (shift posisi). Bukan natural date.
4. **Overnight rolling**: POSITION-based (last W valid entries), bukan calendar-day-based.
5. **Broker**: Baca L0 `broksum_bybroker`, rename kolom + base features via DuckDB SQL, aggregate SUM/AVG ke (date,ticker), UPDATE features_store dengan exact column matching.

## Next Priority

1. Kalibrasi overnight (biar IPCM match production)
2. Fix yf_daily (58 cols, nice to have)
3. CVD + Stockbit + VWAP (terakhir)
