# Edge: bpjs_opening_tp3

## Penjelasan Sederhana

Screener malam hari memilih saham yang **kemungkinan naik di jam pertama trading besok**.
Kita beli pas market buka jam 09:00, keluar jam 10:00. Totalnya cuma 1 jam.

Thesis awalnya: broker aktif di suatu saham → ada aksi → harga ikut naik pagi hari.
Ternyata yang lebih dominan: **saham yang memang karakternya volatile dan sering naik 2%**.
Ini masalahnya — model belajar "siapa yang biasanya naik" bukan "siapa yang besok akan naik".

## Definisi

| Item | Value |
|---|---|
| Strategy | BPJS — Beli Pagi Jual Siang |
| Entry | Open 09:00 WIB (T+1) |
| Exit | Time-based: tepat 10:00 WIB |
| Label TP | `close@10:00 >= open@09:00 * 1.02` (+2%) |
| Label SL | `min_return < -3.5%` |
| Universe | IDX non-blue chip, filter likuiditas minimum |
| Model | LightGBM binary classifier |

## Data Paths

| Path | Keterangan |
|---|---|
| `idx/data/Level_0_Raw/broksum_bybroker.parquet` | Raw broker summary |
| `idx/data/Level_0_Raw/yfinance_1h.parquet` | Intraday 1H |
| `idx/data/Level_0_Raw/yfinance_daily.parquet` | Daily OHLCV |
| `idx/data/Level_1_Features/broksum_datamart.parquet` | Feature L1 |
| `idx/data/Level_2_Datamart/training_datamart_opening_tp3.parquet` | Training table (fixed TP) |
| `idx/data/Level_2_Datamart/training_datamart_opening_rank.parquet` | Training table (relative rank) |

## Scripts

| Script | Command | Output |
|---|---|---|
| `scripts/generate_datamart.py` | `python generate_datamart.py` | fixed TP datamart |
| `scripts/generate_datamart.py` | `python generate_datamart.py --label-mode relative_rank` | relative rank datamart |
| `scripts/train_lightgbm.py` | `/path/.venv/bin/python train_lightgbm.py --output-dir ../../../model/vNN --feature-prune-top-n 0` | `idx/model/vNN/` |

> Jalankan dari `idx/` directory. Pakai `.venv` di root SSSAHAM.

## Feature Families

| Family | Best corr | Status | Catatan |
|---|---|---|---|
| Range/Volatility (`yf_daily_range_*`) | 0.236 | ✅ | Top signal |
| TickerTPRate (`ticker_tp_rate_*`) | 0.224 | ✅ | Historical TP base rate |
| OpenSess (`open_sess_*`) | 0.197 | ✅ | Behavior opening historis |
| Volume raw | 0.100 | ✅ | |
| ADX/DI | 0.096 | ✅ | |
| Stockbit/XL (`xl_*`) | 0.125 | ✅ v17+ | Coverage 18.5%, tidak diangkat model |
| CVD, Broker flow/tfl/ctx | < 0.05 | ⚠️ Noise | |

## Bugs Fixed

| Bug | Fix |
|---|---|
| `relative_volume` dll 100% NULL | ticker `.JK` suffix + `datetime64[ms]` vs `[ns]` mismatch |

## Model Iterations

| Versi | best_iter | AUC OOT | prec@5 | cum_return | Notes |
|---|---|---|---|---|---|
| v1–v12 | — | — | — | — | Iterasi awal |
| v13 | 4 | 0.544 | 0.19 | -9.3% | Baseline |
| v14c | 9 | 0.720 | 0.40 | -8.5% | Feature fix |
| v15d | 10 | 0.719 | 0.38 | **-6.6%** | Best BPJS — SL dilepas |
| v16b | 10 | 0.722 | 0.42 | -16% | Best AUC |
| v17 | 9 | 0.723 | 0.37 | -15.9% | +XL features, tidak membantu |
| v18 | 7 | 0.584 | 0.34 | -13.4% | Relative rank label — AUC turun |

## Confluence & Kesimpulan

**Apa yang berhasil:**
- AUC walkforward stabil di ~0.72 dengan label fixed TP
- Model konsisten memilih saham volatile + historical TP tinggi

**Kenapa tetap gagal:**
- Window 1 jam (09:00–10:00) terlalu sempit — signal historical tidak punya waktu terealisasi
- Break-even win rate ~71% (untuk TP 2%, SL 3.5%, cost 0.4%) — kita hanya mencapai ~42%
- Model belajar **"siapa yang biasanya naik"**, bukan **"siapa yang besok naik"**
- Relative rank label justru lebih susah karena semua feature kita time-series per ticker, bukan cross-sectional

**Area of Growth (kalau mau kembali ke BPJS):**
- **Fix over-regularisasi:** `best_iteration` selalu 4-10 dari 4000 di semua 18 runs — sama seperti BSJP v1b. Root cause: `min_data_in_leaf=500`, `min_gain_to_split=0.1`, `lambda_l1/l2=2.0` terlalu ketat secara bersamaan. Model hampir tidak belajar. Sebelum eksperimen apapun, turunkan dulu ke `--min-data-in-leaf 50 --min-gain-to-split 0.01 --lambda-l1 0.5 --lambda-l2 0.5`.
- **Cross-sectional features:** Butuh perbandingan antar saham pada hari yang sama — fitur yang ada sekarang semua time-series per ticker, bukan cross-sectional
- **Signal lebih forward-looking:** Volatility historis tidak cukup; butuh sinyal yang lebih prediktif untuk window 1 jam

**Rekomendasi:**
BPJS di-pause. Fokus ke BSJP yang sudah PASS. Masalah fundamental (window 1 jam, break-even 71%) tetap ada meski regularisasi di-fix — tapi regularisasi harus di-fix dulu sebelum bisa tahu apakah ada sinyal yang bisa diekstrak.
