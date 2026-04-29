# PRD: Training Datamart (MVP)

## 0) Related PRD

- Dokumen induk: [PRD 0000 — BPJS Screener](./0000_bpjs_screener.md)
- Baseline & feature source: [PRD 0001 — Broker Enrichment](./0001_broker_enrichment.md)
- Layer ini adalah handoff dari Level 1 features ke dataset siap-train.

## 1) What

Menyusun dataset training final untuk objective MVP: prediksi probabilitas `close@10:00` hari T+1 mencapai `>= +3%` dari `open@09:00`, berdasarkan fitur broker+market pada hari T.

## 2) Input / Output Contract

| Item | Value |
|---|---|
| Input features | `data/Level_1_Features/broksum_datamart.parquet` |
| Input market | `data/Level_0_Raw/yfinance_1h.parquet` |
| Input broker master | `data/Level_0_Raw/master_broker.parquet` |
| Label output | `data/Level_2_Datamart/label_opening_to_1000.parquet` |
| Training output | `data/Level_2_Datamart/training_datamart_opening_tp3.parquet` |
| Training grain | `1 row = 1 date(feature_date T) + 1 ticker` |
| Label rule MVP | `close@10:00 >= open@09:00 * 1.03` |
| Cutoff | `10:00 WIB` |

## 3) Readiness Audit (As Of 2026-04-20)

| Check | Value | Status |
|---|---:|---|
| Level 0 broksum max date | 2026-04-17 | OK |
| Level 0 yfinance_daily max date | 2026-04-17 | OK |
| Level 0 yfinance_1h max date | 2026-04-17 | OK |
| Level 1 feature max date | 2026-04-17 | ✅ FRESH — regenerated 2026-04-20 |
| Training max feature date | — | STALE — L2 belum regenerate |
| Label max trade date | — | STALE — L2 belum regenerate |
| Gap L1 vs L0 (hari) | 0 | OK |
| Duplicate key broksum (date/broker/ticker) | 0 | OK |
| broksum rows | 2,909,407 | OK — Oct backfill complete |
| broksum date count | 363 | OK (Oct 2024 → Apr 2026) |
| broksum broker/day (min/mean/max) | 82/84.5/86 | ✅ Oct backfill selesai |
| L1 rows | 2,909,407 | OK |
| L1 columns | 122 | OK — dengan prefix families |
| yfinance dates vs broksum (selisih) | 39 hari | OK — semua IDX holidays |
| TP rate (`label_tp`) | 0.0394 | Info — terlalu rendah, pertimbangkan relabel |
| SL3 rate (`label_sl3`) | 0.0816 | Info — SL > TP, cost roundtrip masalah |

### Data Quality Issues (Updated 2026-04-20)

| Issue | Detail | Status |
|---|---|---|
| `broker_type` = 'Unknown' semua | Diganti dengan Foreign/LocalFund/Retail dari LLM consensus | ✅ RESOLVED — `brkm_broker_type` |
| `buy_sell_val_ratio` null (pure buy rows) | 472K rows gross_sell=0 — null menyembunyikan signal | ✅ RESOLVED — `brkm_is_buy_only` flag + ratio filled 99.0 |
| `avg_spread_pct` null (pure sell rows) | 403K rows gross_buy=0 | ✅ RESOLVED — `brkm_is_sell_only` flag + ratio filled 0.01 |
| Z-score outliers | `yf_daily_range_z_5` max=1.43e+15 | ✅ RESOLVED — semua z-score di-clip ±10 |
| Foreign/domestic classification | master_broker.category null 90/92 broker | ✅ RESOLVED — LLM consensus, 18 foreign / 74 domestic |
| Oct 2024 hanya 28 broker/hari | DATA_DIR bug setelah restructuring | ✅ RESOLVED — backfill 57 broker, sekarang 82–86/hari |
| broker×ticker sparsity | 58.4% pairs < 20 hari → ma_20 NaN | 🟡 OPEN — handle di L2 filter (min obs threshold) |

**Readiness Verdict:** `L1 READY — siap regenerate L2`

## 3.5) Level 2 — Data Points & Feature Engineering Pipeline

Level 2 mengubah grain dari `date/broker/ticker` (L1) menjadi `date/ticker` (1 baris per saham per hari) yang siap masuk model. Proses utama: agregasi broker, enrichment kategori broker, HMM regime, dan labeling.

### Input ke Level 2

| Source | Path | Dipakai Untuk |
|---|---|---|
| L1 broksum datamart | `Level_1_Features/broksum_datamart.parquet` | Semua broker features |
| yfinance 1h | `Level_0_Raw/yfinance_1h.parquet` | Label computation (entry/exit price) |
| yfinance daily | `Level_0_Raw/yfinance_daily.parquet` | IHSG daily untuk HMM regime |
| master_broker | `Level_0_Raw/master_broker.parquet` | Klasifikasi broker (localfund/bandar/foreign) |

### L1 Column Prefix Schema (As Of 2026-04-20)

| Prefix | Keluarga | Contoh Kolom |
|---|---|---|
| *(no prefix)* | Keys & metadata | `broker`, `ticker`, `date`, `scraped_at` |
| `flow_` | Raw + derived flow | `flow_gross_buy`, `flow_net_flow_ratio`, `flow_churn_ratio` |
| `brkm_` | Broker metadata | `brkm_broker_type`, `brkm_is_foreign`, `brkm_is_buy_only` |
| `ctx_` | Context/relative | `ctx_broker_market_share`, `ctx_broker_net_buy_rank` |
| `tfl_` | Temporal flow (rolling) | `tfl_net_buy_ma_5`, `tfl_net_buy_z_5`, `tfl_sign_consistency_5` |
| `act_` | Activity | `act_active_days_in_5`, `act_days_since_last_active` |
| `pc_` | Price context (post-merge) | `pc_avg_buy_price_in_range`, `pc_buy_volume_in_total_volume` |
| `yf_daily_` | Market daily | `yf_daily_close`, `yf_daily_range_z_5` |
| `yf_1h_` | Market 1h | `yf_1h_ret_oc`, `yf_1h_turnover` |
| `yf_4h_` | Market 4h | `yf_4h_range_pct` |

### Grup Fitur di Level 2

#### GRP-A: Broker Aggregate (sum/mean across all brokers per date×ticker)

Setiap kolom L1 di-aggregate dua cara: SUM (total aktivitas semua broker) dan MEAN (rata-rata per broker). Ini menjawab: *"hari ini semua broker di saham X secara total/rata-rata ngapain?"*

| Kolom L1 yang di-aggregate (nama dengan prefix) | Keterangan |
|---|---|
| `flow_total_net_buy` | Net flow uang (buy-sell) seluruh broker di saham ini |
| `flow_gross_turnover` | Total transaksi (beli+jual) seluruh broker |
| `flow_buy_freq`, `flow_sell_freq` | Frekuensi transaksi buy/sell |
| `flow_abs_net_buy` | Absolut net buy (magnitude tanpa arah) |
| `flow_net_flow_ratio` | Rasio net terhadap gross (arah dominan) |
| `flow_total_trades` | Jumlah trade |
| `flow_net_buy_per_trade` | Nilai rata-rata per trade |
| `flow_churn_ratio` | Gross/net — makin tinggi makin scalping |
| `ctx_broker_ticker_specificity` | Seberapa fokus broker ke saham ini vs market |
| `ctx_broker_market_share` | Share broker dari total market |
| `ctx_ticker_market_share` | Share saham dari total market |
| `ctx_broker_net_buy_rank` | Rank percentile broker di antara semua broker (0=terkecil, 1=terbesar) |
| `tfl_net_buy_ma_5/20/60` | Rolling MA net buy (5, 20, 60 hari), clipped ±10 |
| `tfl_net_buy_z_5/20/60` | Z-score net buy vs historis, clipped ±10 |
| `tfl_net_buy_velocity_5/20/60` | Percepatan perubahan net buy |
| `tfl_net_flow_ratio_ma_5/20/60` | Rolling MA net flow ratio |
| `tfl_churn_ratio_ma_5/20/60` | Rolling MA churn ratio |
| `tfl_net_buy_sign_consistency_5` | % hari dengan arah beli yang sama dalam 5 hari terakhir |
| `act_active_days_in_5` | Berapa dari 5 hari terakhir broker aktif di saham ini |
| `act_days_since_last_active` | Hari sejak terakhir kali broker aktif di saham ini |
| `pc_avg_buy_price_in_range` | Posisi avg buy price dalam range hari (0=di lows, 1=di highs) |
| `pc_avg_sell_price_in_range` | Posisi avg sell price dalam range hari |
| `pc_buy_volume_in_total_volume` | Proporsi volume beli broker terhadap total volume pasar |

#### GRP-B: Breadth Features

Menjawab: *"berapa banyak broker aktif, dan apakah lebih banyak buyer atau seller?"*

| Kolom | Keterangan |
|---|---|
| `broker_count` | Jumlah broker yang aktif di saham ini hari ini |
| `buyer_broker_count` | Broker dengan net_buy > 0 |
| `seller_broker_count` | Broker dengan net_buy < 0 |
| `buyer_ratio` | buyer_count / broker_count (0–1, >0.5 = lebih banyak buyer) |
| `seller_ratio` | seller_count / broker_count |

#### GRP-C: Focus Broker Signals (MVP: broker MG)

Sinyal individual broker yang dianggap memiliki predictive power tinggi (MVP: MG = Semesta Indovest). Di-extend nanti untuk broker lain.

| Kolom | Keterangan |
|---|---|
| `mg_total_net_buy` | Nilai net buy broker MG di saham ini hari ini |
| `mg_total_net_buy_z_20` | Z-score net buy MG vs 20 hari terakhir |
| `mg_total_net_buy_velocity_20` | Velocity net buy MG |
| `mg_net_flow_ratio` | Arah dominan MG (buy/sell) |
| `mg_churn_ratio` | Apakah MG scalping atau genuine accumulation |
| `mg_present` | Binary — apakah MG aktif di saham ini hari ini |

#### GRP-D: Localfund Aggregate Signals

Broker yang diklasifikasikan sebagai "local fund" (manajemen investasi lokal, reksa dana domestik). Agregasi perilaku kolektif mereka. Menjawab: *"dana institusional lokal sedang akumulasi atau distribusi?"*

| Kolom | Keterangan |
|---|---|
| `localfund_netbuy_sum/mean` | Total/rata-rata net buy broker localfund |
| `localfund_abs_netbuy_sum` | Magnitude absolut aktivitas localfund |
| `localfund_broker_count` | Berapa localfund aktif hari ini |
| `localfund_buyer/seller_broker_count` | Balance buyer vs seller di localfund |
| `localfund_concentration_hhi` | HHI — apakah satu localfund dominan atau tersebar merata |
| `localfund_top1_share` / `top3_share` | Konsentrasi 1 dan 3 broker terbesar |
| `localfund_streak_buy_days` | Berapa hari berturut-turut localfund net-buy positif |
| `localfund_netbuy_ma20` / `std20` / `z20` | Tren 20 hari + z-score localfund aggregate |
| `localfund_netbuy_3d_sum` / `7d_sum` | Akumulasi 3 hari dan 7 hari |
| `localfund_buy_freq_z30` | Z-score frekuensi beli localfund vs 30 hari |
| `localfund_participation_ratio` | Rasio localfund terhadap total broker aktif |
| `localfund_consensus_strength` | Proxy kekuatan konsensus buyer-seller localfund |

#### GRP-E: Bandar Aggregate Signals

Broker yang secara empiris sering menjadi "market maker" atau "bandar" di saham second-liner. Struktur sama dengan localfund tapi untuk kelompok broker ini.

| Kolom | Keterangan |
|---|---|
| `bandar_netbuy_sum/mean` | Total/rata-rata net buy broker bandar |
| `bandar_abs_netbuy_sum` | Magnitude absolut aktivitas bandar |
| `bandar_broker_count` | Berapa bandar aktif hari ini |
| `bandar_buyer/seller_broker_count` | Balance buyer vs seller di bandar |
| `bandar_concentration_hhi` | Konsentrasi aksi — apakah satu bandar dominan |
| `bandar_top1_share` / `top3_share` | Konsentrasi 1 dan 3 bandar terbesar |
| `bandar_buy_freq_sum/mean` | Total/rata-rata frekuensi transaksi buy |

#### GRP-F: Market Regime (HMM — Hidden Markov Model)

IHSG dikategorikan ke 3 state menggunakan Hidden Markov Model berbasis log-return harian. State diurutkan secara canonical berdasarkan mean return: 0=bear, 1=sideways, 2=bull. Menjawab: *"hari ini market secara keseluruhan sedang di fase apa?"*

| Kolom | Keterangan |
|---|---|
| `ihsg_regime` | State regime: 0=bear, 1=sideways, 2=bull (float32) |
| `ihsg_regime_confidence` | Posterior probability HMM untuk state yang dipilih (0–1) |

**Catatan:** Jika library `hmmlearn` tidak tersedia, fallback ke quantile-based regime (qcut 33/67 percentile). Warmup minimum: 60 hari IHSG history sebelum regime valid.

#### GRP-G: Label / Outcome (T+1)

Data outcome hari T+1 yang di-join ke feature tanggal T. Ini adalah target supervised learning dan data evaluasi backtest.

| Kolom | Keterangan |
|---|---|
| `entry_price_opening` | Harga open T+1 jam 09:00 WIB (entry price) |
| `high_to_cutoff` | High intraday T+1 dari 09:00 sampai 10:00 WIB |
| `low_to_cutoff` | Low intraday T+1 dari 09:00 sampai 10:00 WIB |
| `close_to_cutoff` | Close intraday T+1 pada 10:00 WIB (exit price) |
| `max_return_to_cutoff` | (high / entry) - 1 — potensi TP maksimum |
| `min_return_to_cutoff` | (low / entry) - 1 — potensi SL maksimum |
| `close_return_to_cutoff` | (close / entry) - 1 — return aktual di cutoff |
| `label_tp` | **Target label** — 1 jika close_return >= +3% |
| `label_sl3` | Bitmask stop-loss — 1 jika min_return <= -3% |
| `trade_date` | Tanggal trade T+1 |
| `feature_date` | Tanggal feature T (key join ke training table) |

### Feature Gaps yang Masih Open

| Gap | Prioritas | Keterangan |
|---|---|---|
| broker×ticker sparsity | 🟡 MED | 58.4% pairs < 20 hari → tfl_net_buy_ma_20 NaN. Handle di L2: filter min obs atau impute |
| Foreign broker aggregate (L2) | 🟡 MED | `brkm_is_foreign` sudah ada di L1, tapi aggregate sinyal asing (sum/mean net buy) belum di-compute di L2 |
| `tfl_churn_ratio_std/z` | 🟢 LOW | churn_ratio temporal tidak punya std/z, tidak konsisten dengan tfl_net_buy_* |

## 4) Training Datamart Summary

| Metric | Value |
|---|---:|
| L0 broksum rows | 2,909,407 |
| L0 broksum date range | 2024-10-01 → 2026-04-17 (363 dates) |
| L0 broksum broker/day | 82–86 (post-backfill) |
| L1 datamart rows | 2,909,407 |
| L1 datamart columns | 122 (dengan prefix families) |
| L1 date range | 2024-10-01 → 2026-04-17 |
| L1 regenerated at | 2026-04-20 |
| L2 training rows | — (belum regenerate) |
| L2 training columns | — (belum regenerate) |

### 4.1 Feature Groups (Training Table)

| Group | Contoh Kolom | Tujuan |
|---|---|---|
| Identity | `date`, `ticker` | Key observasi training. |
| Broker Aggregate | `*_sum`, `*_mean` | Ringkasan statistik broker-level dari Level 1 pada (date,ticker). |
| Breadth | `broker_count`, `buyer_ratio` | Keseimbangan partisipasi broker buyer/seller. |
| Focus Broker (MG) | `mg_total_net_buy`, `mg_present` | Sinyal broker fokus (MVP). |
| Localfund Signals | `localfund_*` | Agregat perilaku localfund (streak, z-score, concentration). |
| Bandar Signals | `bandar_*` | Agregat perilaku broker kategori bandar. |
| Label Join | `entry_price_opening`, `close_to_cutoff`, `label_tp` | Outcome T+1 untuk supervised learning. |

## 5) Full Column Dictionary — Training Table

| Column | Type | Group | Definition |
|---|---|---|---|
| `date` | `timestamp[us]` | Identity | Feature date (T). |
| `ticker` | `large_string` | Identity | Ticker symbol at daily grain. |
| `total_net_buy_sum` | `double` | Broker Aggregate | Agregasi SUM dari kolom L1 `total_net_buy` across brokers di (date,ticker). |
| `total_net_buy_mean` | `double` | Broker Aggregate | Agregasi MEAN dari kolom L1 `total_net_buy` across brokers di (date,ticker). |
| `gross_turnover_sum` | `double` | Broker Aggregate | Agregasi SUM dari kolom L1 `gross_turnover` across brokers di (date,ticker). |
| `gross_turnover_mean` | `double` | Broker Aggregate | Agregasi MEAN dari kolom L1 `gross_turnover` across brokers di (date,ticker). |
| `buy_freq_sum` | `int64` | Broker Aggregate | Agregasi SUM dari kolom L1 `buy_freq` across brokers di (date,ticker). |
| `buy_freq_mean` | `double` | Broker Aggregate | Agregasi MEAN dari kolom L1 `buy_freq` across brokers di (date,ticker). |
| `sell_freq_sum` | `int64` | Broker Aggregate | Agregasi SUM dari kolom L1 `sell_freq` across brokers di (date,ticker). |
| `sell_freq_mean` | `double` | Broker Aggregate | Agregasi MEAN dari kolom L1 `sell_freq` across brokers di (date,ticker). |
| `abs_net_buy_sum` | `double` | Broker Aggregate | Agregasi SUM dari kolom L1 `abs_net_buy` across brokers di (date,ticker). |
| `abs_net_buy_mean` | `double` | Broker Aggregate | Agregasi MEAN dari kolom L1 `abs_net_buy` across brokers di (date,ticker). |
| `net_flow_ratio_sum` | `double` | Broker Aggregate | Agregasi SUM dari kolom L1 `net_flow_ratio` across brokers di (date,ticker). |
| `net_flow_ratio_mean` | `double` | Broker Aggregate | Agregasi MEAN dari kolom L1 `net_flow_ratio` across brokers di (date,ticker). |
| `total_trades_sum` | `int64` | Broker Aggregate | Agregasi SUM dari kolom L1 `total_trades` across brokers di (date,ticker). |
| `total_trades_mean` | `double` | Broker Aggregate | Agregasi MEAN dari kolom L1 `total_trades` across brokers di (date,ticker). |
| `net_buy_per_trade_sum` | `double` | Broker Aggregate | Agregasi SUM dari kolom L1 `net_buy_per_trade` across brokers di (date,ticker). |
| `net_buy_per_trade_mean` | `double` | Broker Aggregate | Agregasi MEAN dari kolom L1 `net_buy_per_trade` across brokers di (date,ticker). |
| `churn_ratio_sum` | `double` | Broker Aggregate | Agregasi SUM dari kolom L1 `churn_ratio` across brokers di (date,ticker). |
| `churn_ratio_mean` | `double` | Broker Aggregate | Agregasi MEAN dari kolom L1 `churn_ratio` across brokers di (date,ticker). |
| `broker_ticker_specificity_sum` | `double` | Broker Aggregate | Agregasi SUM dari kolom L1 `broker_ticker_specificity` across brokers di (date,ticker). |
| `broker_ticker_specificity_mean` | `double` | Broker Aggregate | Agregasi MEAN dari kolom L1 `broker_ticker_specificity` across brokers di (date,ticker). |
| `broker_market_share_sum` | `double` | Broker Aggregate | Agregasi SUM dari kolom L1 `broker_market_share` across brokers di (date,ticker). |
| `broker_market_share_mean` | `double` | Broker Aggregate | Agregasi MEAN dari kolom L1 `broker_market_share` across brokers di (date,ticker). |
| `ticker_market_share_sum` | `double` | Broker Aggregate | Agregasi SUM dari kolom L1 `ticker_market_share` across brokers di (date,ticker). |
| `ticker_market_share_mean` | `double` | Broker Aggregate | Agregasi MEAN dari kolom L1 `ticker_market_share` across brokers di (date,ticker). |
| `total_net_buy_ma_5_sum` | `double` | Broker Aggregate | Agregasi SUM dari kolom L1 `total_net_buy_ma_5` across brokers di (date,ticker). |
| `total_net_buy_ma_5_mean` | `double` | Broker Aggregate | Agregasi MEAN dari kolom L1 `total_net_buy_ma_5` across brokers di (date,ticker). |
| `total_net_buy_z_5_sum` | `double` | Broker Aggregate | Agregasi SUM dari kolom L1 `total_net_buy_z_5` across brokers di (date,ticker). |
| `total_net_buy_z_5_mean` | `double` | Broker Aggregate | Agregasi MEAN dari kolom L1 `total_net_buy_z_5` across brokers di (date,ticker). |
| `total_net_buy_velocity_5_sum` | `double` | Broker Aggregate | Agregasi SUM dari kolom L1 `total_net_buy_velocity_5` across brokers di (date,ticker). |
| `total_net_buy_velocity_5_mean` | `double` | Broker Aggregate | Agregasi MEAN dari kolom L1 `total_net_buy_velocity_5` across brokers di (date,ticker). |
| `total_net_buy_ma_20_sum` | `double` | Broker Aggregate | Agregasi SUM dari kolom L1 `total_net_buy_ma_20` across brokers di (date,ticker). |
| `total_net_buy_ma_20_mean` | `double` | Broker Aggregate | Agregasi MEAN dari kolom L1 `total_net_buy_ma_20` across brokers di (date,ticker). |
| `total_net_buy_z_20_sum` | `double` | Broker Aggregate | Agregasi SUM dari kolom L1 `total_net_buy_z_20` across brokers di (date,ticker). |
| `total_net_buy_z_20_mean` | `double` | Broker Aggregate | Agregasi MEAN dari kolom L1 `total_net_buy_z_20` across brokers di (date,ticker). |
| `total_net_buy_velocity_20_sum` | `double` | Broker Aggregate | Agregasi SUM dari kolom L1 `total_net_buy_velocity_20` across brokers di (date,ticker). |
| `total_net_buy_velocity_20_mean` | `double` | Broker Aggregate | Agregasi MEAN dari kolom L1 `total_net_buy_velocity_20` across brokers di (date,ticker). |
| `total_net_buy_ma_60_sum` | `double` | Broker Aggregate | Agregasi SUM dari kolom L1 `total_net_buy_ma_60` across brokers di (date,ticker). |
| `total_net_buy_ma_60_mean` | `double` | Broker Aggregate | Agregasi MEAN dari kolom L1 `total_net_buy_ma_60` across brokers di (date,ticker). |
| `total_net_buy_z_60_sum` | `double` | Broker Aggregate | Agregasi SUM dari kolom L1 `total_net_buy_z_60` across brokers di (date,ticker). |
| `total_net_buy_z_60_mean` | `double` | Broker Aggregate | Agregasi MEAN dari kolom L1 `total_net_buy_z_60` across brokers di (date,ticker). |
| `total_net_buy_velocity_60_sum` | `double` | Broker Aggregate | Agregasi SUM dari kolom L1 `total_net_buy_velocity_60` across brokers di (date,ticker). |
| `total_net_buy_velocity_60_mean` | `double` | Broker Aggregate | Agregasi MEAN dari kolom L1 `total_net_buy_velocity_60` across brokers di (date,ticker). |
| `net_flow_ratio_ma_5_sum` | `double` | Broker Aggregate | Agregasi SUM dari kolom L1 `net_flow_ratio_ma_5` across brokers di (date,ticker). |
| `net_flow_ratio_ma_5_mean` | `double` | Broker Aggregate | Agregasi MEAN dari kolom L1 `net_flow_ratio_ma_5` across brokers di (date,ticker). |
| `net_flow_ratio_ma_20_sum` | `double` | Broker Aggregate | Agregasi SUM dari kolom L1 `net_flow_ratio_ma_20` across brokers di (date,ticker). |
| `net_flow_ratio_ma_20_mean` | `double` | Broker Aggregate | Agregasi MEAN dari kolom L1 `net_flow_ratio_ma_20` across brokers di (date,ticker). |
| `net_flow_ratio_ma_60_sum` | `double` | Broker Aggregate | Agregasi SUM dari kolom L1 `net_flow_ratio_ma_60` across brokers di (date,ticker). |
| `net_flow_ratio_ma_60_mean` | `double` | Broker Aggregate | Agregasi MEAN dari kolom L1 `net_flow_ratio_ma_60` across brokers di (date,ticker). |
| `churn_ratio_ma_5_sum` | `double` | Broker Aggregate | Agregasi SUM dari kolom L1 `churn_ratio_ma_5` across brokers di (date,ticker). |
| `churn_ratio_ma_5_mean` | `double` | Broker Aggregate | Agregasi MEAN dari kolom L1 `churn_ratio_ma_5` across brokers di (date,ticker). |
| `churn_ratio_ma_20_sum` | `double` | Broker Aggregate | Agregasi SUM dari kolom L1 `churn_ratio_ma_20` across brokers di (date,ticker). |
| `churn_ratio_ma_20_mean` | `double` | Broker Aggregate | Agregasi MEAN dari kolom L1 `churn_ratio_ma_20` across brokers di (date,ticker). |
| `churn_ratio_ma_60_sum` | `double` | Broker Aggregate | Agregasi SUM dari kolom L1 `churn_ratio_ma_60` across brokers di (date,ticker). |
| `churn_ratio_ma_60_mean` | `double` | Broker Aggregate | Agregasi MEAN dari kolom L1 `churn_ratio_ma_60` across brokers di (date,ticker). |
| `has_yf_daily_max` | `int8` | Coverage | 1 jika data daily market tersedia pada panel ticker-hari itu. |
| `has_yf_1h_max` | `int8` | Coverage | 1 jika data 1H market tersedia pada panel ticker-hari itu. |
| `has_yf_4h_max` | `int8` | Coverage | 1 jika data 4H market tersedia pada panel ticker-hari itu. |
| `broker_count` | `int64` | Breadth | Jumlah broker aktif di (date,ticker). |
| `buyer_broker_count` | `int32` | Breadth | Jumlah broker dengan `total_net_buy > 0`. |
| `seller_broker_count` | `int32` | Breadth | Jumlah broker dengan `total_net_buy < 0`. |
| `buyer_ratio` | `double` | Breadth | buyer_broker_count / broker_count. |
| `seller_ratio` | `double` | Breadth | seller_broker_count / broker_count. |
| `mg_total_net_buy` | `double` | Focus Broker (MG) | Nilai net buy broker MG di ticker-hari itu. |
| `mg_total_net_buy_z_20` | `double` | Focus Broker (MG) | Z-score 20D broker MG dari feature source L1. |
| `mg_total_net_buy_velocity_20` | `double` | Focus Broker (MG) | Velocity 20D broker MG dari feature source L1. |
| `mg_net_flow_ratio` | `double` | Focus Broker (MG) | Net flow ratio broker MG. |
| `mg_churn_ratio` | `double` | Focus Broker (MG) | Churn ratio broker MG. |
| `mg_present` | `int8` | Focus Broker (MG) | 1 jika broker MG muncul di ticker-hari itu. |
| `localfund_netbuy_sum` | `double` | Localfund Signals | Agregasi SUM untuk `localfund_netbuy` dari broker localfund pada (date,ticker). |
| `localfund_netbuy_mean` | `double` | Localfund Signals | Agregasi MEAN untuk `localfund_netbuy` dari broker localfund pada (date,ticker). |
| `localfund_abs_netbuy_sum` | `double` | Localfund Signals | Agregasi SUM untuk `localfund_abs_netbuy` dari broker localfund pada (date,ticker). |
| `localfund_broker_count` | `int32` | Localfund Signals | Fitur agregat/komposit localfund untuk ticker-hari tersebut. |
| `localfund_buyer_broker_count` | `int32` | Localfund Signals | Fitur agregat/komposit localfund untuk ticker-hari tersebut. |
| `localfund_seller_broker_count` | `int32` | Localfund Signals | Fitur agregat/komposit localfund untuk ticker-hari tersebut. |
| `localfund_buy_freq_sum` | `double` | Localfund Signals | Agregasi SUM untuk `localfund_buy_freq` dari broker localfund pada (date,ticker). |
| `localfund_buy_freq_mean` | `double` | Localfund Signals | Agregasi MEAN untuk `localfund_buy_freq` dari broker localfund pada (date,ticker). |
| `localfund_concentration_hhi` | `double` | Localfund Signals | Fitur agregat/komposit localfund untuk ticker-hari tersebut. |
| `localfund_top1_share` | `double` | Localfund Signals | Fitur agregat/komposit localfund untuk ticker-hari tersebut. |
| `localfund_top3_share` | `double` | Localfund Signals | Fitur agregat/komposit localfund untuk ticker-hari tersebut. |
| `bandar_netbuy_sum` | `double` | Bandar Signals | Agregasi SUM untuk `bandar_netbuy` dari broker bandar pada (date,ticker). |
| `bandar_netbuy_mean` | `double` | Bandar Signals | Agregasi MEAN untuk `bandar_netbuy` dari broker bandar pada (date,ticker). |
| `bandar_abs_netbuy_sum` | `double` | Bandar Signals | Agregasi SUM untuk `bandar_abs_netbuy` dari broker bandar pada (date,ticker). |
| `bandar_broker_count` | `int32` | Bandar Signals | Fitur agregat/komposit bandar untuk ticker-hari tersebut. |
| `bandar_buyer_broker_count` | `int32` | Bandar Signals | Fitur agregat/komposit bandar untuk ticker-hari tersebut. |
| `bandar_seller_broker_count` | `int32` | Bandar Signals | Fitur agregat/komposit bandar untuk ticker-hari tersebut. |
| `bandar_buy_freq_sum` | `double` | Bandar Signals | Agregasi SUM untuk `bandar_buy_freq` dari broker bandar pada (date,ticker). |
| `bandar_buy_freq_mean` | `double` | Bandar Signals | Agregasi MEAN untuk `bandar_buy_freq` dari broker bandar pada (date,ticker). |
| `bandar_concentration_hhi` | `double` | Bandar Signals | Fitur agregat/komposit bandar untuk ticker-hari tersebut. |
| `bandar_top1_share` | `double` | Bandar Signals | Fitur agregat/komposit bandar untuk ticker-hari tersebut. |
| `bandar_top3_share` | `double` | Bandar Signals | Fitur agregat/komposit bandar untuk ticker-hari tersebut. |
| `localfund_participation_ratio` | `double` | Localfund Signals | Rasio broker localfund terhadap total broker. |
| `localfund_buyer_ratio` | `double` | Localfund Signals | Rasio localfund buyer terhadap broker buyer. |
| `localfund_consensus_strength` | `double` | Localfund Signals | Kekuatan konsensus localfund (proxy buyer-seller balance). |
| `localfund_streak_buy_days` | `int32` | Localfund Signals | Streak hari berturut localfund net-buy positif. |
| `localfund_netbuy_ma20` | `double` | Localfund Signals | MA20 netbuy localfund aggregate. |
| `localfund_netbuy_std20` | `double` | Localfund Signals | STD20 netbuy localfund aggregate. |
| `localfund_netbuy_z20` | `double` | Localfund Signals | Z-score netbuy localfund vs MA20/STD20. |
| `localfund_netbuy_3d_sum` | `double` | Localfund Signals | Akumulasi netbuy localfund 3 hari. |
| `localfund_netbuy_7d_sum` | `double` | Localfund Signals | Akumulasi netbuy localfund 7 hari. |
| `localfund_buy_freq_ma30` | `double` | Localfund Signals | MA30 buy frequency localfund. |
| `localfund_buy_freq_std30` | `double` | Localfund Signals | STD30 buy frequency localfund. |
| `localfund_buy_freq_z30` | `double` | Localfund Signals | Z-score buy frequency localfund vs MA30/STD30. |
| `localfund_buy_freq_ratio_to_ma30` | `double` | Localfund Signals | Rasio buy frequency saat ini terhadap MA30. |
| `entry_price_opening` | `double` | Label Join | Harga open entry T+1 (09:00). |
| `high_to_cutoff` | `double` | Label Join | High intraday sampai cutoff (10:00). |
| `low_to_cutoff` | `double` | Label Join | Low intraday sampai cutoff (10:00). |
| `close_to_cutoff` | `double` | Label Join | Close intraday pada cutoff (10:00). |
| `volume` | `int64` | Label Join | Volume pada candle cutoff. |
| `trade_date` | `timestamp[ms]` | Label Join | Tanggal trade T+1. |
| `entry_datetime` | `timestamp[us]` | Label Join | Timestamp entry open (09:00). |
| `entry_hm` | `int32` | Label Join | Kode jam-menit entry (HHMM). |
| `bars_until_cutoff` | `int64` | Label Join | Jumlah bar dari entry ke cutoff (MVP=1). |
| `max_return_to_cutoff` | `double` | Label Join | (high_to_cutoff / entry_price_opening) - 1. |
| `min_return_to_cutoff` | `double` | Label Join | (low_to_cutoff / entry_price_opening) - 1. |
| `close_return_to_cutoff` | `double` | Label Join | (close_to_cutoff / entry_price_opening) - 1. |
| `label_tp` | `int64` | Label Join | Target label binary TP (+3% sampai 10:00). |
| `label_sl3` | `int64` | Label Join | Stop-loss label binary -3% sampai 10:00. |
| `label_name` | `large_string` | Label Join | Nama konfigurasi label. |

## 6) Full Column Dictionary — Label Table

| Column | Type | Group | Definition |
|---|---|---|---|
| `ticker` | `large_string` | Identity | Ticker symbol. |
| `entry_price_opening` | `double` | Price/Return | Harga open entry. |
| `high_to_cutoff` | `double` | Price/Return | Harga high sampai cutoff. |
| `low_to_cutoff` | `double` | Price/Return | Harga low sampai cutoff. |
| `close_to_cutoff` | `double` | Price/Return | Harga close di cutoff. |
| `volume` | `int64` | Price/Return | Volume candle cutoff. |
| `trade_date` | `timestamp[ms]` | Identity | Tanggal trade (T+1). |
| `entry_datetime` | `timestamp[us]` | Entry/Cutoff | Timestamp entry open (09:00). |
| `entry_hm` | `int32` | Entry/Cutoff | Kode jam-menit entry. |
| `bars_until_cutoff` | `int64` | Entry/Cutoff | Jumlah bar dari entry ke cutoff (MVP=1). |
| `max_return_to_cutoff` | `double` | Price/Return | Return maksimum sampai cutoff. |
| `min_return_to_cutoff` | `double` | Price/Return | Return minimum sampai cutoff. |
| `close_return_to_cutoff` | `double` | Price/Return | Return close di cutoff. |
| `label_tp` | `int64` | Label | 1 jika close_return_to_cutoff >= +3%. |
| `label_sl3` | `int64` | Label | 1 jika min_return_to_cutoff <= -3%. |
| `feature_date` | `timestamp[ms]` | Identity | Tanggal fitur (T) yang dipakai untuk prediksi. |

## 7) Acceptance Criteria (Ready to Train)

- Freshness: `max(training.date)` sama dengan `max(Level1.date)`.
- Key integrity: duplicate key training = 0, duplicate key label = 0.
- Coverage: `has_yf_daily_max`, `has_yf_1h_max`, `has_yf_4h_max` tersedia dan coverage memadai.
- Label integrity: `label_tp` binary (`0/1`) dan objective sesuai PRD MVP (+3% sampai 10:00).
- Joinability: training sudah mengandung kolom label/outcome yang dibutuhkan untuk modeling baseline.
