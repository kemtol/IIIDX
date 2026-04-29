# Test Case: Create Level 1 Feature Mart (`broksum_datamart.parquet`)

## 1. Tujuan
Memvalidasi proses pembuatan **Level 1 feature mart** dari bahan **Level 0**:
- `broksum_bybroker.parquet`
- `yfinance_daily.parquet`
- `yfinance_1h.parquet`
- `yfinance_4h.parquet`
- `master_emiten.parquet`

Output target:
- `data/Level_1_Features/broksum_datamart.parquet`

## 2. Scope
- Script utama: `pipeline/feature/generate_broksum_datamart.py`
- Runner: `pipeline/run/run_feature_l1.sh`
- Fokus test: readiness input, dry-run generation, dan status freshness output L1.

## 3. Current Alignment (As-Is, 2026-04-14)

### 3.1 Status Level 0 (Bahan)
Semua sumber Level 0 tersedia dan terisi:
- `broksum_bybroker.parquet`: max `2026-04-13`
- `yfinance_daily.parquet`: max `2026-04-14`
- `yfinance_1h.parquet`: max `2026-04-14 14:00`
- `yfinance_4h.parquet`: max `2026-04-14 13:00`
- `master_emiten.parquet`: tersedia (773 row)

Catatan: hasil ini align dengan progress test `0001_daily_fetcher.md` (Level 0 fetch sudah jalan).

### 3.2 Status Level 1 Existing
Snapshot existing `broksum_datamart.parquet`:
- rows: `346,336`
- date range: `2026-02-02` s.d. `2026-04-13`

Interpretasi:
- File Level 1 sudah ter-rebuild dan **sudah mengejar freshness broksum** sampai `2026-04-13`.
- Catatan operasional saat ini: rebuild dijalankan dengan `--date-from 2026-02-01` untuk menghindari bottleneck I/O full-history.

## 4. Test Scenarios

### TC-L1-001 Input Readiness Check
- Objective: memastikan semua input L0 untuk generator L1 ada.
- Expected:
1. Semua file source tersedia.
2. Kolom waktu utama (`date`/`datetime`) bisa dibaca.

### TC-L1-002 Dry-Run Feature Generation (Sampling)
- Objective: memastikan pipeline feature engineering L1 berjalan end-to-end tanpa write final.
- Command:
```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  pipeline/feature/generate_broksum_datamart.py \
  --skip-refresh-level0 \
  --sample-rows 5000 \
  --dry-run
```
- Expected:
1. Exit code `0`.
2. Stage pipeline selesai sampai `Stage4-MarketMerged`.
3. Total column output = `106`.

### TC-L1-003 Full Build Level 1 (No Refresh)
- Objective: menghasilkan output Level 1 terbaru dari snapshot Level 0 saat ini.
- Command:
```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  pipeline/feature/generate_broksum_datamart.py \
  --skip-refresh-level0
```
- Expected:
1. Exit code `0`.
2. File output `broksum_datamart.parquet` ter-write ulang.
3. `max(date)` Level 1 minimal mengejar `max(date)` broksum source (2026-04-13).

### TC-L1-004 Output Sanity
- Objective: validasi kualitas minimal output L1.
- Checks:
1. Duplicate key (`date`,`broker`,`ticker`) = 0.
2. Kolom kunci wajib ada: `date`, `broker`, `ticker`, `total_net_buy`, `net_flow_ratio`.
3. Coverage flag tidak semuanya nol: `has_yf_daily`, `has_yf_1h`, `has_yf_4h`.

## 5. Quick Validation Queries
```python
import pandas as pd

l1 = pd.read_parquet("data/Level_1_Features/broksum_datamart.parquet")
print("shape:", l1.shape)
print("date range:", pd.to_datetime(l1["date"]).min(), pd.to_datetime(l1["date"]).max())
print("dup key:", l1.duplicated(subset=["date", "broker", "ticker"]).sum())

for c in ["has_yf_daily", "has_yf_1h", "has_yf_4h"]:
    if c in l1.columns:
        print(c, l1[c].value_counts(dropna=False).to_dict())
```

## 6. Hasil Eksekusi Saat Ini (2026-04-14)

| Test Case | Status | Evidence |
|---|---|---|
| TC-L1-001 Input Readiness | PASS | Semua file Level 0 tersedia dan terbaca |
| TC-L1-002 Dry-Run Sampling | PASS | Pipeline selesai, total kolom `106` |
| TC-L1-003 Full Build No Refresh | PASS | Full build operasional selesai, output write sukses |
| TC-L1-004 Output Sanity | PASS | Duplicate key `0`, coverage market terisi, freshness `max(date)=2026-04-13` |

Detail dry-run sampling terakhir:
- Source snapshot:
  - broksum max date: `2026-04-13`
  - yf daily max date: `2026-04-14`
  - yf 1h max datetime: `2026-04-14 14:00`
  - yf 4h max datetime: `2026-04-14 13:00`
- Stage output:
  - `Stage4-MarketMerged` rows: `5,000`
  - total columns: `106`
  - coverage:
    - `has_yf_daily`: `95.26%`
    - `has_yf_1h`: `95.26%`
    - `has_yf_4h`: `95.26%`

Detail full-build operasional terakhir:
- Command:
```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  pipeline/feature/generate_broksum_datamart.py \
  --skip-refresh-level0 \
  --date-from 2026-02-01
```
- Result:
  - `Stage4-MarketMerged` rows: `346,336`
  - total columns: `106`
  - date range: `2026-02-02` -> `2026-04-13`
  - coverage:
    - `has_yf_daily`: `90.47%`
    - `has_yf_1h`: `88.13%`
    - `has_yf_4h`: `88.13%`

## 7. Kesimpulan Alignment
Asumsi kamu **benar** dan align dengan progress:
1. Bahan Level 0 sudah siap.
2. Pipeline Level 1 sudah terbukti jalan.
3. Level 1 saat ini siap dipakai lanjut ke datamart training.

## 8. Next Action
1. Generate ulang datamart training (`/data/Level_2_Datamart`) dari Level 1 terbaru.
2. Audit readiness train (freshness, duplicate key, coverage, label rate).
3. Dokumentasikan kontrak datamart training di folder PRD.
