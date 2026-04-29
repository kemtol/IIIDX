# Test Case: Daily Fetcher (`fetch_yfinance_daily.py`)

## 1. Tujuan
Memvalidasi bahwa daily fetcher:
- menarik data OHLCV harian dari Yahoo Finance untuk ticker IDX,
- melakukan update incremental (bukan duplikasi),
- aman dijalankan berulang (idempotent),
- menghasilkan output Level 0 yang valid untuk pipeline downstream.

## 2. Scope
- Script: `pipeline/fetch/fetch_yfinance_daily.py`
- Input utama: `data/Level_0_Raw/master_emiten.parquet`
- Output utama: `data/Level_0_Raw/yfinance_daily.parquet`
- Eksekusi: manual CLI + scheduler (cron heartbeat)

## 3. Non-Scope
- Akurasi harga market vs broker feed lain (cross-vendor reconciliation)
- Strategi model/training
- Integritas data intraday (`1h`/`4h`)

## 4. Pre-Condition
1. Python env tersedia (`.venv`) dan dependency `pandas`, `pyarrow`, `yfinance` terinstall.
2. File `master_emiten.parquet` tersedia dan minimal punya kolom `ticker`.
3. Internet outbound aktif (akses ke Yahoo Finance).
4. User punya izin write ke folder `data/Level_0_Raw/`.

## 5. Post-Condition
1. File output parquet terbentuk / terupdate.
2. Tidak ada row duplikat pada key (`date`, `ticker`).
3. Skema output tetap konsisten:
   - `date`, `ticker`, `open`, `high`, `low`, `close`, `volume`

## 6. Test Data Setup
- Gunakan master emiten real.
- Untuk test cepat gunakan `--limit-tickers 10`.
- Untuk test produksi gunakan tanpa limit.

## 7. Test Scenarios

### TC-001 Happy Path (Initial Build)
- Objective: memastikan fetch pertama berhasil membuat file output valid.
- Command:
```bash
python pipeline/fetch/fetch_yfinance_daily.py \
  --master-path data/Level\ 0/master_emiten.parquet \
  --output data/Level\ 0/yfinance_daily.parquet \
  --limit-tickers 10
```
- Expected:
1. Proses selesai exit code `0`.
2. Output parquet terbentuk.
3. Kolom wajib ada semua.
4. Row count `> 0`.

### TC-002 Incremental Update (No Full Rebuild)
- Objective: memastikan run kedua hanya upsert data baru/overlap, tidak reset total.
- Steps:
1. Catat `rows_before` dan `max_date_before`.
2. Jalankan command yang sama.
3. Catat `rows_after` dan `max_date_after`.
- Expected:
1. `rows_after >= rows_before`.
2. `max_date_after >= max_date_before`.
3. Tidak terjadi lonjakan aneh akibat duplikasi massal.

### TC-003 Idempotency (Run Berulang Hari Sama)
- Objective: memastikan run berulang pada hari yang sama tidak menghasilkan duplikasi key.
- Steps:
1. Jalankan script 2x berturut-turut.
2. Query duplicate count by (`date`, `ticker`).
- Expected:
1. Duplicate count = `0`.
2. Row growth minimal/0 jika tidak ada data baru.

### TC-004 ACTIVE Filter Default
- Objective: memastikan default mode hanya memproses emiten aktif (jika kolom `status` ada).
- Command:
```bash
python pipeline/fetch/fetch_yfinance_daily.py --limit-tickers 20
```
- Expected:
1. Script tidak error walau `status` tidak ada.
2. Jika `status` ada, ticker non-active tidak diprioritaskan (sesuai filter default).

### TC-005 All Status Override
- Objective: memastikan override `--all-status` bekerja.
- Command:
```bash
python pipeline/fetch/fetch_yfinance_daily.py \
  --all-status \
  --limit-tickers 20
```
- Expected:
1. Eksekusi sukses.
2. Kandidat ticker tidak dibatasi status active saja.

### TC-006 Missing Master Emiten File
- Objective: memastikan error handling jelas saat input master tidak ada.
- Command:
```bash
python pipeline/fetch/fetch_yfinance_daily.py \
  --master-path data/Level\ 0/not_exists.parquet
```
- Expected:
1. Exit non-zero.
2. Pesan error eksplisit `master emiten parquet not found`.

### TC-007 Invalid Master Schema
- Objective: memastikan validasi kolom `ticker` berjalan.
- Setup:
1. Buat parquet dummy tanpa kolom `ticker`.
- Expected:
1. Exit non-zero.
2. Error jelas: kolom `ticker` tidak ditemukan.

### TC-008 Output Write Permission Denied
- Objective: memastikan kegagalan write terdeteksi.
- Setup:
1. Arahkan `--output` ke lokasi tanpa izin tulis.
- Expected:
1. Exit non-zero.
2. Error I/O jelas.

### TC-009 Cron Wrapper Integration
- Objective: memastikan wrapper cron memanggil parameter path Level 0 yang benar.
- Script:
`pipeline/run/run_fetch_yfinance_daily.sh`
- Expected:
1. Path `--master-path` mengarah ke `data/Level_0_Raw/master_emiten.parquet`.
2. Path `--output` mengarah ke `data/Level_0_Raw/yfinance_daily.parquet`.
3. Exit code `0` saat dijalankan manual.

### TC-010 Data Quality Sanity
- Objective: validasi kualitas data dasar setelah fetch.
- Checks:
1. `date` ter-parse sebagai tanggal valid.
2. `ticker` uppercase dan non-empty.
3. `volume > 0` untuk semua row.
4. `open/high/low/close` non-null untuk row valid.
- Expected:
1. Tidak ada pelanggaran fatal.

## 8. Quick Validation Queries
```python
import pandas as pd

df = pd.read_parquet("data/Level_0_Raw/yfinance_daily.parquet")
print(df.shape)
print(df.columns.tolist())
print(df["date"].min(), df["date"].max())

dup = df.duplicated(subset=["date", "ticker"]).sum()
print("duplicate(date,ticker) =", dup)

print("null_ohlc =", df[["open","high","low","close"]].isna().sum().to_dict())
print("non_positive_volume =", (df["volume"] <= 0).sum())
```

## 9. Exit Criteria (MVP)
Semua kondisi berikut harus terpenuhi:
1. TC-001, TC-002, TC-003, TC-009 lulus.
2. Duplicate key (`date`,`ticker`) = 0.
3. Output schema sesuai kontrak.
4. Cron wrapper dapat dijalankan tanpa ubah parameter manual.

## 10. Catatan
- Daily fetcher ini adalah sumber data dasar (Level 0), jadi prioritas utamanya stabilitas dan idempotency, bukan kecepatan ekstrim.
- Jika ada mismatch tanggal terbaru, cek urutan scheduler `master_emiten -> yfinance_daily`.

## 11. Hasil Eksekusi Test (2026-04-14)

Environment:
- cwd: ``
- python: `.venv/bin/python` (`Python 3.12.3`)
- timezone: `Asia/Jakarta`

Ringkasan hasil:

| Test Case | Status | Evidence Singkat |
|---|---|---|
| TC-001 Happy Path | PASS | Exit `0`, output valid, `rows=1,832,011`, kolom lengkap 7 field |
| TC-002 Incremental Update | PASS | Before `1832011|2026-04-14`, after `1832011|2026-04-14`, tidak ada reset |
| TC-003 Idempotency | PASS | Run 2x berturut, `duplicate(date,ticker)=0`, rows tetap |
| TC-004 ACTIVE Filter Default | PASS | Exit `0`, run sukses dengan default filter (`--limit-tickers 20`) |
| TC-005 All Status Override | PASS | Exit `0`, run sukses dengan `--all-status` |
| TC-006 Missing Master File | PASS | Exit `1`, error: `master emiten parquet not found` |
| TC-007 Invalid Master Schema | PASS | Exit `1`, error: `Column 'ticker' not found...` |
| TC-008 Output Permission Denied | PASS | Exit `1`, error `PermissionError` saat output `/root/...` |
| TC-009 Cron Wrapper Integration | PASS | `run_fetch_yfinance_daily.sh` exit `0`, path input/output Level 0 benar |
| TC-010 Data Quality Sanity | FAIL | `non_positive_volume=218907` (expected `0`) |

Detail metrik utama dari run:
- `yfinance_daily.parquet`
  - `rows`: `1,832,011`
  - `columns`: `['date','ticker','open','high','low','close','volume']`
  - `date range`: `2011-06-03` s.d. `2026-04-14`
  - `duplicate(date,ticker)`: `0`
  - `null OHLC`: `0`
  - `non_positive_volume`: `218,907` (temuan)

Kesimpulan sementara:
1. Fungsionalitas fetcher incremental + idempotent sudah stabil.
2. Error handling input/output sudah sesuai ekspektasi.
3. Temuan kualitas data pada historical volume (`<=0`) perlu remediation sebelum dipakai strict QA gate.

Rekomendasi tindak lanjut untuk temuan TC-010:
1. Jalankan one-time cleanup untuk drop row `volume <= 0` di `yfinance_daily.parquet`.
2. Tambahkan guard saat merge existing data agar baris legacy `volume <= 0` ikut tersaring saat write.
