# Edge: bsjp_overnight_sl2

## Penjelasan Sederhana

Bayangkan kamu beli saham **sore hari sekitar jam 15:30**, lalu jual **besok pagi jam 09:05** pas market baru buka.

Pertanyaannya: saham mana yang kemungkinan besar harganya lebih tinggi besok pagi dibanding kemarin sore?

Model menemukan tiga sinyal utama yang bekerja:

1. **Saham yang sore itu aktif bergerak naik** — momentum 1 jam terakhir sebelum close (14:xx→15:xx) dan range hari ini lebar = ada buyer aktif, cenderung lanjut gap up overnight
2. **Saham yang punya track record sering gap up** — kalau dalam 20-60 hari terakhir saham ini konsisten naik dari sore ke pagi, polanya cenderung berulang
3. **Saham yang historisnya JARANG gap down parah** — saham dengan track record overnight_p10 yang baik (tidak pernah -10%+ dalam 5 hari terakhir) jauh lebih aman dipilih

Singkatnya: **"ride the overnight momentum pada saham yang aktif tapi tidak volatile ke bawah, exit sebelum retail penuh masuk"**.

> Poin 3 adalah temuan baru di v7 — model sekarang bisa membedakan saham "momentum bagus" vs saham yang sering kena gap down tiba-tiba.

## Definisi

| Item | Value |
|---|---|
| | Strategy | BSJP Normal — Beli Sore Jual Pagi |
| | Entry | Close ~15:30–15:45 WIB (proxy: close candle jam 15:xx hari T) |
| | Exit | Time-based: tepat 09:05 WIB T+1 (proxy: open candle jam 09:xx) |
| | Label | `overnight_return = (open@09:xx_T+1 - close@15:xx_T) / close@15:xx_T` |
| | Label positif | `overnight_return > roundtrip_cost (0.4%)` |
| | SL | `overnight_return < -2%` (klasifikasi saja, bukan cut intraday — market tutup) |
| | Universe | IDX non-blue chip, second-liner |
| | Model | LightGBM binary classifier |
| | Policy | Top-3 per hari, skewed weight 60-30-10 (rank #1 dapat 60% modal) |

## No-Lookahead Rules

| Data | Status | Keterangan |
|---|---|---|
| | OHLCV close candle 15:xx hari T | ✅ OK | Entry price proxy |
| | Broksum T-1 (shift 1) | ✅ OK | Aman, pakai hari sebelumnya |
| | Gap down history T-1 ke T-N | ✅ OK | Rolling window dari masa lalu |
| | OHLCV open candle 09:xx T+1 | ❌ Label only | Exit price, JANGAN jadi feature |

## Data Paths

| Path | Keterangan |
|---|---|
| | `idx/data/Level_0_Raw/broksum_bybroker.parquet` | Raw broker summary |
| | `idx/data/Level_0_Raw/yfinance_1h.parquet` | Intraday 1H — entry & exit price |
| | `idx/data/Level_1_Features/broksum_datamart.parquet` | Feature L1 |
| | `idx/data/Level_2_Datamart/training_datamart_bsjp_overnight.parquet` | Training table (234 cols) |

## Scripts

| Script | Command | Output |
|---|---|---|
| | `scripts/generate_datamart.py` | `python generate_datamart.py` | L2 datamart |
| | `scripts/train_lightgbm.py` | `python train_lightgbm.py --output-dir model/bsjp_vNN --feature-prune-top-n 0 --tp-pct 0.01 --sl-pct -0.02 --oot-valid-days 100 --rank-weights "0.6,0.3,0.1"` | `idx/model/bsjp_vNN/` |

> Jalankan dari `idx/` directory. Pakai `.venv` di root SSSAHAM.

## Feature Families (v7 → v15, top features dari 249)

> Update v15: IHSG MA features masuk top-30. `close_range_pct` (13,514) naik ke #2. Gap down profile gain turun dari 898 → 0 (tidak lagi top-30) karena IHSG MA context menggantikan fungsi downside protection. Lihat [feature importance v15](../model/BSJP/bsjp_v15/feature_importance.csv).

| Family | Relevansi | Top Feature | Gain | Catatan |
|---|---|---|---|---|
| | **Closing momentum** | ⭐⭐⭐⭐⭐ | `close_ret_last1h` | 24,681 | Return 14:xx→15:xx + range hari ini — DOMINAN mutlak |
| | **Overnight history (gap up)** | ⭐⭐⭐⭐ | `overnight_positive_rate60` | 7,974 | Persistensi gap up 20-60 hari — karakter saham |
| | **Daily OHLCV** | ⭐⭐⭐ | `close_vs_open_day` | 6,004 | Intraday range + volume velocity |
| | **Global macro** | ⭐⭐⭐ | `vix_prev_close` | 1,793 | VIX + USDIDR — Wall Street sentiment |
| | **IHSG MA context** *(baru v15)* | ⭐⭐ | `ihsg_close_ma20_ratio` | 870 | IHSG MA5/20/100/200 + prev_return — konteks trend makro |
| | **Gap down profile** *(v7)* | ⭐ | *(tidak top-30)* | 0 | Digantikan IHSG MA context — fungsi downside protection sudah tercakup |
| | Broker aggregate (L1) | ⭐ | minor | <500 | Minor contribution |
| | Stockbit/XL activity | — | — | 0 | Tidak masuk top features |
| | CVD | — | — | 0 | Tidak masuk top features |

### Kenapa Gap Down Profile Masuk?

Setiap saham punya "karakter" downside yang berbeda. GTRA, CSIS, PANI secara historis lebih sering gap down ekstrem dari saham lain. Model kini bisa membaca sinyal ini dari:
- `overnight_p10_5d` — jika 10th percentile 5-hari terakhir sangat negatif, saham ini "berbahaya" saat ini
- `overnight_worst_5d` — worst overnight dalam 5 hari terakhir
- `gapdown_freq_20d` / `gapdown_severe_freq_20d` — frekuensi gap down dalam 20 hari

Window pendek (5d, 20d) yang relevan — karakter gap down bersifat *recent*, bukan historis panjang (60d importance = 0).

## Model Iterations

> **Catatan unit**: `cum_return` di tabel ini adalah hasil simulasi portofolio OOT (100 hari), dihitung sebagai equity compounded minus 1 — bukan simple sum return. 1.15 = +115%, 32.56 = +3257%.

### v1–v7: Overnight Objective (Entry close T, Exit open T+1 ~09:05)

| Versi | OOT days | AUC OOT | cum_return | max_dd | Notes |
|---|---|---|---|---|---|
| | bsjp_v1b | 20 | 0.495 | +14.79% | -0.14% | PASS awal (underfit, OOT terlalu pendek) |
| | bsjp_v3 | 60 | 0.601 | +48.16% | -12.46% | Unlock deep model |
| | bsjp_v4 | 100 | 0.601 | +417.55% | -15.33% | SL clip bug — overstate |
| | v4_honest | 100 | 0.601 | +78.84% | -57.44% | Realitas v4 tanpa SL clip |
| | bsjp_v5 | 100 | 0.570 | -82.44% | -92.89% | FAIL — universe filter buang sumber edge |
| | bsjp_v6 | 100 | 0.601 | +79.73% | -56.47% | SL clip fix, baseline |
| | **bsjp_v7** | **100** | **0.602** | **+102.85%** | **-30.19%** | **BEST overnight saat dievaluasi — gap down features + skewed weight 60-30-10** |

> **Penting**: angka v7 di atas adalah dari evaluasi OOT window *lama*. Ketika di-run ulang dengan OOT window terkini (100 hari s.d. Apr 2026), v7 hanya menghasilkan +1.0% — karena OOT window sudah geser ke periode berbeda.

### v8–v11: Overnight + Eksperimen Post-v7

Semua run di sini pakai datamart overnight yang sama (`training_datamart_bsjp_overnight.parquet`), OOT 100 hari s.d. Apr 2026.

| Versi | AUC OOT | cum_return | max_dd | Notes |
|---|---|---|---|---|
| | bsjp_v8 | 0.592 | +0.7% | -40.6% | Explorasi post-v7, lebih buruk |
| | bsjp_v9 | 0.594 | +0.3% | -32.0% | Explorasi, tidak improve |
| | bsjp_v9b_repro_v7_cut | 0.596 | +1.0% | -22.8% | Sanity check — v7 berhasil direproduksi |
| | bsjp_v9c_ytd | 0.578 | -0.4% | -66.9% | Year-to-date eval — konfirmasi 2026 lebih keras |
| | bsjp_v9d_ranker | 0.589 | -0.0% | -47.4% | LGBMRanker — GAGAL (win rate 0.44) |
| | **bsjp_v11_hyperopt** | **0.595** | **+1668%** | **-38.9%** | Hyperopt longgarkan regulasi — hasil paper sangat tinggi, vol harian 14.9%/hari |

> **Soal v11_hyperopt**: return +1668% secara matematis konsisten dengan mean daily net 3.79% compounded 100 hari, tapi vol harian 14.9% mengindikasikan model memilih saham yang sangat volatile. Perlu Monte Carlo dan paper trade sebelum dipercaya.

### v12–v15: BSJP Overnight + IHSG MA Features (Apr 2026)

Setelah v11, fokus kembali ke **overnight objective** dengan penambahan **IHSG moving average features** (MA5/20/100/200) untuk konteks trend makro. Datamart diperbarui ke 730 hari.

| Versi | AUC OOT | CumRet | MaxDD | Vol/hari | Win Rate | Params (md/λ/gain) | Notes |
|-------|---------|--------|-------|----------|----------|-------------------|-------|
| bsjp_v12 | 0.598 | +1326% | -57.2% | 13.6% | 65% | md=100, λ=1.0, gain=0.1 | Old datamart (728 days), best_iter=38 |
| bsjp_v13 | 0.587 | +638% | -40.3% | 10.6% | 66% | md=100, λ=1.0, gain=0.1 | Refreshed datamart (730 days), best_iter=24 |
| **bsjp_v15** | **0.602** | **+974%** | **-15.3%** | **7.9%** | **68%** | **md=50, λ=0.5, gain=0.01** | **Best DD — IHSG MA features, best_iter=27** |
| bsjp_v15_1 | 0.602 | +974% | -15.3% | 7.9% | 68% | md=50, λ=0.5, gain=0.01 | Repro ✅ identik |

> **v15 Highlights:** MaxDD -15.3% adalah **terendah dari SEMUA model BSJP** — turun drastis dari -30.2% (v7) dan -57.2% (v12). IHSG MA features memberikan konteks makro yang mencegah model mengambil posisi saat trend IHSG memburuk. Vol harian 7.9% juga paling rendah — setara ~1.3% per jam vs saham second-liner yang bisa bergap -5% s.d. +10%.
>
> **Monte Carlo v15 ✅** (10k paths, block bootstrap 5d): P(terminal < 0) = **0.0%** di kedua horizon 100d dan 252d. Mean MaxDD -17.3% (100d) / -20.4% (252d). P(MaxDD ≤ -30%) hanya 0.67% untuk 100d. Hasil ini mengonfirmasi v15 sebagai model BSJP **paling robust secara statistik**. Lihat [Monte Carlo v15](../model/BSJP/bsjp_v15/monte_carlo/monte_summary_metrics.csv).

> **⚠️ PERINGATAN DATA LEAKAGE (Apr 2026):** Semua model close10 di bawah ini (`v10_close10_*`, grid search, `v11_hyperopt_close10_*`) dilatih menggunakan `training_datamart_bsjp_close10_v10.parquet` yang mengandung **target leakage**: kolom `exit_price`, `overnight_return`, dan `close_ret_last1h` (tidak diblokir oleh OUTCOME_COLS, digunakan sebagai feature #1). Lihat [LATEST.md Bagian 5F](../model/BSJP/LATEST.md#5f--data-leakage--invalidasi-model-close10_v10-apr-2026). **Status: INVALID** — perlu di-re-run dengan datamart clean.

### v10 → Grid: Close10 Objective (Entry open T+1 ~09:00, Exit close T+1 ~10:00)

Sejak v10, terbentuk track baru: **objective diubah** dari overnight ke intraday 1 jam pagi (entry open 09:00, exit close 10:00). Datamart baru: `training_datamart_bsjp_close10_v10.parquet`.

| Versi | Params (md/λ) | Policy | AUC OOT | cum_return | max_dd | vol/hari | Sharpe | Notes |
|---|---|---|---|---|---|---|---|---|
| | bsjp_v10_close10_obj | md=500, λ=2.0 | k=3, 60-30-10 | 0.577 | +226.5% | -48.5% | 5.74% | 3.72 | Objective baru |
| | bsjp_v10_close10_obj_k2_w25_guard | md=500, λ=2.0 | k=2, max_w=0.25 | 0.577 | **+115.2%** | **-26.0%** | 3.69% | 3.57 | **Kandidat institusional** — Monte Carlo valid |
| | bsjp_v11_hyperopt (overnight orig) | — | k=3, 60-30-10 | 0.595 | +1668% | -38.9% | 14.9% | 4.03 | Hyperopt, vol sangat tinggi |
| | bsjp_v11_hyperopt_close10_k2w25 | — | k=2, max_w=0.25 | — | **+329.8%** | **-52.5%** | — | 2.51 | Hyperopt di close10, DD dalam |

## Grid Search — Full Report (Apr 2026)

Grid search dilakukan untuk mencari sweet spot `min_data_in_leaf` (md) dan `lambda_l1`/`lambda_l2` (lam) pada LightGBM dengan objective close10. Dua fase: **coarse** (16 kombinasi, md × lam) dilanjutkan **fine** (28 kombinasi, md × lam × gain).

**Common settings:** Policy threshold, p_cut=0.035, max_positions=3, max_weight=0.34, rank_weights "0.6,0.3,0.1". Datamart: 112,386 rows, 728 hari (2023-03-07 → 2026-04-21), OOT 100 hari.

### Phase 1: Coarse Grid (md × lam, gain=0.05 fixed)

Grid: `md ∈ [50, 100, 200, 500]`, `lam ∈ [0.1, 0.5, 1.0, 2.0]`

```
Dir            md  lam  AUC_OOT  CumRet%  MaxDD%  Sharpe  Gap_AUC  BestIter
-------------------------------------------------------------------------------
md100_lam1_0  100  1.0   0.5863   2531.7   -31.3   4.733   0.0816      125
md200_lam1_0  200  1.0   0.5896   2140.1   -26.2   4.438   0.0781      139
md100_lam2_0  100  2.0   0.5864   2132.9   -32.0   4.643   0.0703       85
md050_lam2_0   50  2.0   0.5740   2033.7   -39.7   4.476   0.0749       47
md050_lam1_0   50  1.0   0.5750   1711.5   -55.9   3.850   0.0745       49
md100_lam0_5  100  0.5   0.5863   1304.0   -39.5   3.802   0.0712       84
md100_lam0_1  100  0.1   0.5857    963.3   -39.5   3.489   0.0724       85
md050_lam0_5   50  0.5   0.5770    938.5   -48.5   3.536   0.0727       49
md200_lam2_0  200  2.0   0.5874    867.9   -49.9   3.372   0.0735      114
md050_lam0_1   50  0.1   0.5862    736.5   -42.4   3.249   0.0688       69
md500_lam1_0  500  1.0   0.5831    662.4   -51.5   3.431   0.0613       68
md200_lam0_5  200  0.5   0.5878    592.7   -53.4   3.009   0.0744      116
md200_lam0_1  200  0.1   0.5873    511.8   -55.7   2.864   0.0627       69
md500_lam2_0  500  2.0   0.5825    386.1   -50.7   2.824   0.0620       69
md500_lam0_5  500  0.5   0.5833    352.9   -51.2   2.710   0.0612       68
md500_lam0_1  500  0.1   0.5838    338.8   -58.5   2.670   0.0607       69
```

**Key Findings (Coarse):**
- **md=100 adalah sweet spot** — dominating top-3 by CumRet, Sharpe terbaik.
- **md=500 (v7 default) adalah terburuk** — CumRet cuma 339-662%, konfirmasi over-regularization.
- **lam=1.0 optimal** di kedua md=100 dan md=200. lam terlalu rendah (0.1, 0.5) atau terlalu tinggi (2.0) performanya turun.
- **md=50 volatile** — return tinggi (2034%) tapi MaxDD -55.9%, indikasi overfit.
- **Overfit Gap** lebih rendah di md besar (500: 0.060-0.062) vs md kecil (50: 0.069-0.075), konsisten dengan teori regularisasi.

### Phase 2: Fine Grid (md × lam × gain)

Grid: `md ∈ [80, 100, 150, 200]`, `lam ∈ [0.75, 1.0, 1.5, 2.0]`, `gain ∈ [0.02, 0.05, 0.1]` — 48 combos planned, 28 executed (selective gain testing: full gain grid only for promising md/lam combos).

```
Dir                  md  lam  gain  AUC_OOT  CumRet%  MaxDD%  Sharpe  Gap_AUC  BestIter
-----------------------------------------------------------------------------------------------
md100_lam1_5_gain0_02  100  1.5  0.02   0.5869   3442.9   -31.6   4.993   0.0814      128
md100_lam1_5_gain0_05  100  1.5  0.05   0.5869   3442.9   -31.6   4.993   0.0814      128
md100_lam1_0_gain0_1   100  1.0   0.1   0.5843   3256.6   -28.2   4.923   0.0729       85
md100_lam1_5_gain0_1   100  1.5   0.1   0.5854   3181.9   -26.0   5.109   0.0713       85
md080_lam1_5_gain0_05   80  1.5  0.05   0.5852   2585.1   -24.1   4.786   0.0680       69
md100_lam1_0_gain0_05  100  1.0  0.05   0.5863   2531.7   -31.3   4.733   0.0816      125
md200_lam1_5_gain0_05  200  1.5  0.05   0.5878   2145.8   -52.2   4.090   0.0739      116
md200_lam1_5_gain0_02  200  1.5  0.02   0.5880   2145.3   -52.2   4.089   0.0738      116
md200_lam1_0_gain0_05  200  1.0  0.05   0.5896   2140.1   -26.2   4.438   0.0781      139
md100_lam2_0_gain0_02  100  2.0  0.02   0.5864   2132.9   -32.0   4.643   0.0703       85
md200_lam1_5_gain0_1   200  1.5   0.1   0.5878   2009.8   -40.8   4.250   0.0740      116
md080_lam1_0_gain0_05   80  1.0  0.05   0.5846   1610.3   -33.7   4.234   0.0692       69
md080_lam0_75_gain0_05  80  0.75 0.05   0.5744   1274.7   -43.3   3.867   0.0741       47
md100_lam0_75_gain0_05 100  0.75 0.05   0.5825   1174.0   -44.9   3.663   0.0708       69
... (28 combos total, 0 failures)
```

**Key Findings (Fine):**
- **md=100 + lam=1.5 adalah optimum global** — CumRet +3443%, Sharpe 4.99, BestIter 128.
- **min_gain_to_split hampir tidak berdampak** — gain=0.02, 0.05, 0.1 menghasilkan CumRet identik untuk md/lam yang sama (beda <0.1%). Parameter ini bisa diabaikan.
- **md=100 konsisten unggul** di semua lam value. md=80 kompetitif (+2585%) tapi md=150-200 mulai menurun drastis.
- **Sharpe terbaik: md=100, lam=1.5, gain=0.1** — Sharpe 5.109 (vs 4.993 untuk gain=0.02). Trade-off: CumRet lebih rendah (+3182% vs +3443%) tapi MaxDD lebih dangkal (-26.0% vs -31.6%).
- **Overfit Gap tertinggi** (±0.081) justru pada best performer — wajar karena model lebih kompleks (md lebih kecil = lebih banyak leaf).
- **Best iteration** (early stopping) berkisar 47-139. Best performer berhenti di 128 — masih reasonable, tidak terlalu cepat.

### Comparison: Historical Models vs Grid Search Best

| Model | CumRet | MaxDD | Sharpe | Vol/hari | AUC OOT | Notes |
|---|---|---|---|---|---|---|
| v10_gapchar | +9% | -50.4% | 0.65 | — | — | Overnight original, gagal |
| v10_close10_obj | +226% | -48.5% | 3.72 | 5.7% | 0.577 | Baseline close10 |
| v10_close10_k2_w25_guard | **+115%** | **-26.0%** | **3.57** | **3.7%** | 0.577 | **Kandidat institusional** |
| v11_hyperopt (overnight) | +1668% | -38.9% | 4.03 | 14.9% | 0.595 | Vol sangat tinggi |
| v11_hyperopt_close10_k2w25 | +330% | -52.5% | 2.51 | — | — | DD dalam |
| **Grid: md100_lam1_5 (fine)** | **+3443%** | **-31.6%** | **4.99** | **~14%** | **0.587** | **Best paper return** |
| **Grid: md100_lam1_0_gain0_1** | **+3257%** | **-28.2%** | **4.92** | **13.9%** | **0.584** | **Sharpe tinggi, DD dangkal** |

> **⚠️ Peringatan:** Semua grid result di atas adalah **paper returns** — belum divalidasi Monte Carlo. Vol harian 13-14% menghasilkan angka compounded yang sangat besar secara matematis (mean daily 4.3% × 100 hari = ribuan persen), tapi belum tentu realistik secara empiris. Overfit risk tinggi karena regulasi sangat longgar (md=100, lam=1.5).

### Recommended Candidates

**1. Best Risk-Adjusted (Grid):** `md100_lam1_5_gain0_1`
   - Sharpe 5.109 (tertinggi dari 44 combos), CumRet +3182%, MaxDD -26.0%
   - Gain=0.1 memberikan regularisasi tambahan yang menekan DD.
   - BestIter 85 — lebih efisien dari 128.

**2. Best Absolute Return (Grid):** `md100_lam1_5_gain0_02`
   - CumRet +3443%, Sharpe 4.993, MaxDD -31.6%
   - Return tertinggi, tapi DD lebih dalam.

**3. Best Conservative (Institutional):** `v10_close10_obj_k2_w25_guard`
   - Sharpe 3.57, CumRet +115%, MaxDD -26%, vol 3.7%/hari
   - Satu-satunya dengan validasi Monte Carlo: P(terminal < 0) = 2.70%.

### Scripts & Artefak

- Grid search coarse: [`edges/bsjp_overnight_sl2/scripts/grid_search_coarse.sh`](edges/bsjp_overnight_sl2/scripts/grid_search_coarse.sh)
- Grid search fine: [`edges/bsjp_overnight_sl2/scripts/grid_search_fine.sh`](edges/bsjp_overnight_sl2/scripts/grid_search_fine.sh)
- Collector: [`model/BSJP/bsjp_grid_coarse/collect_results.py`](model/BSJP/bsjp_grid_coarse/collect_results.py)
- Full results: [`model/BSJP/bsjp_grid_coarse/grid_run.log`](model/BSJP/bsjp_grid_coarse/grid_run.log), [`model/BSJP/bsjp_grid_fine/grid_search_fine.log`](model/BSJP/bsjp_grid_fine/grid_search_fine.log)

## Lessons Learned

### ✅ SL Clip Bug — FIXED di v6+
Market tutup 15:30→09:00, tidak ada kesempatan cut intraday. `simulate_portfolio` kini pakai raw `overnight_return`.

### ❌ Universe Filter — FAILED (v5)
Filter illiquid (turnover < Rp500M) buang 56% data → edge collapse. Saham illiquid/retail-heavy *adalah* sumber edge BSJP. Filter OFF by default, tersedia via `--enable-universe-filter`.

### ❌ Diversifikasi k=10 — Tidak Efektif
Max DD identik (-56% vs -56%), return anjlok. Crash bersifat idiosyncratic (1 saham gap down -20%+), bukan sistemik. Diversifikasi dalam universe sama tidak membantu.

### ❌ Macro/VIX Filter — Dead End
VIX di 10 hari terburuk: rata-rata 17.9 vs 17.3 normal. Hampir identik. Nasdaq mayoritas hijau di hari crash. Root cause bukan macro — tapi karakter saham spesifik.

### ✅ Gap Down Profile — WORKS (v7)
Fitur `overnight_p10_5d`, `overnight_worst_5d`, `gapdown_freq_20d` masuk top-20 features. Max DD turun dari -56% ke **-30%**, return naik dari +80% ke **+103%** (pada OOT window saat v7 dievaluasi).

### ✅ Skewed Weight 60-30-10 — WORKS (v7)
Rank #1 punya TP rate 68% vs rank #2-3 ~52%. Equal weight menyia-nyiakan keunggulan rank #1. Skewed weight sekaligus naikkan return dan turunkan risiko.

### ❌ LGBMRanker — FAILED (v9d)
Dicoba sebagai pengganti binary classifier, lebih aligned ke task top-k. Hasilnya: win rate 0.44, return negatif. Binary classifier lebih stabil.

### ✅ Close10 Objective — SIGNIFICANT IMPROVEMENT
Mengubah exit dari open T+1 (~09:05) ke close T+1 (~10:00) meningkatkan realized upside secara material. Hit rate TP 1% = 69%, dengan mean winner return ~6.4% di sesi pagi. Kandidat terbaik dengan conservative sizing: `v10_close10_obj_k2_w25_guard` — Sharpe 3.57, Sortino 5.74, cum_ret +115%, max_dd -26%.

### ⚠️ Over-Regularization (v7 params) — DITEMUKAN POST-v9
v7 memakai `min_data_in_leaf=500`, `lambda_l2=2.0` — terlalu conservative. Hyperopt (v11) dan grid search menemukan bahwa `md=100`, `λ=1.0-1.5` menghasilkan AUC lebih tinggi dan return paper jauh lebih besar. Namun: vol harian ikut naik 3x (dari ~4% ke ~14%), sehingga angka return compounded menjadi sangat besar dan perlu validasi Monte Carlo sebelum dipercaya.

### ✅ min_gain_to_split — DIABAIKAN (Grid fine)
Variasi gain=0.02/0.05/0.1 tidak menghasilkan perbedaan material pada metrik apapun. Parameter ini bisa di-skip di eksperimen selanjutnya.

### ⚠️ Return Tinggi ≠ Langsung Dipercaya
Grid fine `md100_lam1_5_gain0_02` menunjukkan +3443% OOT di atas kertas, tapi vol harian ~14%. Angka ini secara matematis konsisten (4.4% mean daily compounded 100 hari = ribuan persen), tapi belum ada Monte Carlo validation dan overfit risk tinggi karena regulasi sangat longgar.

### ✅ v15 — IHSG MA Features Berhasil Turunkan Drawdown
Penambahan IHSG MA features (MA5/20/100/200 + prev_return) memberikan konteks makro yang tidak dimiliki model sebelumnya. v15 mencapai MaxDD -15.3% — **terendah sepanjang sejarah BSJP** — tanpa mengorbankan AUC (0.602, setara v7). IHSG features tidak dominan secara individual (gain 500-870), tapi memberikan sinyal **kapan TIDAK boleh trading** yang efektif.

### ✅ Data Integrity — Clean Datamart Terverifikasi (Apr 2026)
Setelah aborted slow build pada 25 Apr 2026, data integrity diverifikasi:
- DuckDB `features_store`: 113,263 vectors, 730 hari (2023-03-07 → 2026-04-23) ✅
- Parquet training datamart: identik 113,263 rows ✅
- IHSG features (MA5/20/100/200) sudah masuk di kedua store ✅
- No missing dates, no duplicate rows ✅
- Benchmark: DuckDB fetch 3.35s, scoring 2.58s (total < 6s) ✅

## Confluence & Kesimpulan

**Tiga track aktif saat ini (per Apr 2026):**

**Track A — Close10, Conservative (Kandidat Institusional) ⚠️ DATA LEAKAGE:**
- `bsjp_v10_close10_obj_k2_w25_guard`
- Entry: open T+1 09:00, Exit: close T+1 10:00
- +115.2% OOT (100 hari), max_dd -26%, Sharpe 3.57
- Vol harian 3.69% — manageable
- Monte Carlo 10k paths: P(terminal < 0) = 2.70%
- **⚠️ Terkontaminasi data leakage** — perlu di-re-run dengan datamart clean

**Track B — Grid Fine, Aggressive ⚠️ DATA LEAKAGE:**
- `bsjp_grid_fine/md100_lam1_5_gain0_02` (Best CumRet: +3443%)
- `bsjp_grid_fine/md100_lam1_5_gain0_1` (Best Sharpe: 5.109)
- Entry: close T 15:30, Exit: open T+1 09:05 (overnight original, tapi datamart close10)
- Paper return 2500-3400% OOT, max_dd -26% to -32%, vol harian ~14%
- **⚠️ Terkontaminasi data leakage + belum Monte Carlo**

**Track C — BSJP Overnight, Conservative ✅ CLEAN (v15):**
- `bsjp_v15` (AUC 0.602, CumRet +974%, MaxDD -15.3%, Vol 7.9%)
- Entry: close T 15:30, Exit: open T+1 09:05
- MaxDD -15.3% = **terendah semua model BSJP**
- Dilatih dengan datamart **clean** (`training_datamart_bsjp_overnight.parquet`) — tidak terkontaminasi data leakage
- IHSG MA features memberikan konteks makro — model tahu kapan tidak boleh trading
- **Monte Carlo ✅** P(terminal < 0) = 0.0% (100d & 252d), mean MaxDD -17.3% (100d)
- **Siap untuk paper trade**

**Key Grid Insight (Close10):** `md=100, lam=1.0-1.5` adalah sweet spot global. md=500 (v7 default) adalah over-regularized. min_gain_to_split tidak berdampak. Semua grid result perlu di-re-run dengan datamart clean sebelum dianggap valid.

## Next Steps

1. **Paper trade v15** — jalankan setiap sore menggunakan [bsjp inference v15](../inferences/bsjp/variants/v15.py), track hasil manual 1-2 minggu
2. **Re-run grid search with clean datamart** — setelah data integrity terverifikasi, grid search perlu diulang dengan `training_datamart_bsjp_overnight.parquet` yang clean (bukan close10_v10 yang terkontaminasi)
3. **Re-run `v10_close10_obj_k2_w25_guard` with clean datamart** — candidate institusional paling menjanjikan, perlu validasi ulang tanpa data leakage
4. **Update REPORT.md** — tambahkan v15 results dan bandingkan Sharpe/Calmar vs track A/B
