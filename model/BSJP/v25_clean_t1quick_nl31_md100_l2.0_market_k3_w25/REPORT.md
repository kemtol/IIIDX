# Laporan Analisis BSJP v25

Evaluasi Baseline vs Half-Kelly untuk Trader Ritel

Tanggal laporan: 21 Mei 2026  
Model: `v25_clean_t1quick_nl31_md100_l2.0_market_k3_w25`  
Objective: `close10` - beli sore hari T, jual sekitar open 10:00 hari T+1

## 1. Ringkasan Eksekutif

Laporan ini membandingkan dua cara memakai model BSJP v25:

| Versi | Cara Pakai | Return 100D OOT | Max Drawdown | Modal Rp10 juta menjadi | Cocok untuk siapa |
|---|---:|---:|---:|---:|---|
| Baseline | 1.0x policy asli | +95.59% | -35.95% | Rp19.56 juta | Trader agresif tapi masih cash-disciplined |
| Half-Kelly | 1.89x baseline | +197.59% | -58.06% | Rp29.76 juta | Trader sangat agresif, paham risiko leverage |

Kesimpulan utama:

- Model baseline sudah punya edge positif: modal OOT Rp10 juta menjadi sekitar Rp19.56 juta dalam 100 trading day.
- Half-Kelly menaikkan return menjadi hampir 3x modal awal, tetapi drawdown historis ikut membesar sampai -58.06%.
- Half-Kelly di sini bukan "setengah modal". Half-Kelly berarti mengalikan return stream baseline sebesar `1.8888x`.
- Secara praktis, baseline jauh lebih realistis untuk eksekusi cash-only. Half-Kelly literal membutuhkan gross exposure sekitar 141.7%, alias sudah masuk area leverage/margin.
- Concern utama model ini bukan sekadar drawdown. Concern utama tetap: no-lookahead, data freshness, inference readiness, dan robustness setelah OOT.

## 2. Cara Kerja Model Dalam Bahasa Awam

Model ini setiap sore memindai saham IDX dan mencari saham yang punya peluang lebih tinggi untuk naik keesokan paginya. Strateginya:

1. Beli sekitar menjelang penutupan hari T.
2. Jual keesokan hari sekitar open 10:00.
3. Ambil maksimal 3 saham.
4. Baseline membatasi bobot maksimal 25% per saham.

Fitur paling dominan berasal dari kondisi intraday sebelum jam keputusan:

| Ranking | Feature | Arti awam |
|---:|---|---|
| 1 | `pre14_tick_pct` | Perkiraan biaya/spread relatif terhadap harga saham |
| 2 | `pre14_range_pct` | Seberapa aktif/fluktuatif saham sampai jam 14 |
| 3 | `pre14_vwap` | Harga rata-rata berbobot volume sebelum sore |
| 4 | `pre14_spread_cost_est` | Estimasi biaya masuk-keluar karena spread |
| 5 | `pre14_pm_turnover` | Nilai transaksi sesi siang |

Plain read: model ini bukan murni pemburu ARA. Ia lebih banyak membaca struktur intraday, biaya transaksi, aktivitas saham, dan pola sebelum entry.

## 3. Apa Itu Half-Kelly di Laporan Ini?

Kelly adalah rumus sizing untuk menentukan seberapa agresif kita boleh menaikkan exposure berdasarkan distribusi return historis.

Dalam laporan ini Kelly dihitung dari return harian portofolio baseline, bukan dari probability per saham.

Hasil perhitungan:

| Item | Nilai |
|---|---:|
| Approx full Kelly (`mean / variance`) | 3.5678x |
| Empirical full Kelly | 3.7776x |
| Half-Kelly | 1.8888x |
| Worst-day ruin limit historis | 9.4990x |

Implikasi sizing:

| Sizing | Max weight per saham | Total gross jika 3 saham |
|---|---:|---:|
| Baseline | 25.00% | 75.00% |
| Half-Kelly literal | 47.22% | 141.66% |

Catatan penting: half-Kelly literal tidak cash-only. Jika tidak mau leverage, scaling maksimal praktis adalah sekitar `1.333x` dari baseline, karena `75% x 1.333 = 100% gross`.

## 4. Perbandingan Performa 100 Hari OOT

Window OOT terkunci: 26 November 2025 sampai 6 Mei 2026.

| Metrik | Baseline | Half-Kelly |
|---|---:|---:|
| Cumulative return | +95.59% | +197.59% |
| Final capital dari Rp10 juta | Rp19.56 juta | Rp29.76 juta |
| Max drawdown | -35.95% | -58.06% |
| Ulcer Index | 15.06% | 26.22% |
| Sharpe annualized | 2.65 | 2.65 |
| Calmar Ratio | 12.30 | 25.17 |
| Win days | 50 / 100 | 50 / 100 |
| Best day | +13.61% | +25.71% |
| Worst day | -10.53% | -19.88% |

Interpretasi:

- Sharpe sama karena half-Kelly hanya menskalakan return harian baseline. Return dan volatilitas naik bersama-sama.
- Calmar terlihat naik karena annualized return meningkat sangat besar, tetapi ini jangan dibaca sebagai "lebih aman". Drawdown absolut half-Kelly jauh lebih dalam.
- Half-Kelly secara mental jauh lebih berat. Penurunan -58% berarti modal Rp10 juta sempat bisa terlihat seperti Rp4.2 juta dari puncak.

Visual:

- `kelly_baseline_vs_half_equity_100d.png`
- `kelly_baseline_vs_half_drawdown_100d.png`

## 5. Rolling Window: 7D, 14D, 30D, 90D

Bagian ini menjawab pertanyaan: "Kalau masuk di periode terakhir, rasanya seperti apa?"

| Window terakhir | Baseline Return | Baseline MaxDD | Half-Kelly Return | Half-Kelly MaxDD |
|---:|---:|---:|---:|---:|
| 7 trading day | -1.88% | -4.19% | -3.72% | -7.84% |
| 14 trading day | -22.13% | -21.54% | -38.57% | -37.68% |
| 30 trading day | +22.31% | -25.64% | +40.86% | -43.77% |
| 90 trading day | +49.78% | -35.95% | +83.37% | -58.06% |
| 100 trading day | +95.59% | -35.95% | +197.59% | -58.06% |

Interpretasi awam:

- Periode 14 hari terakhir sangat buruk. Ini menunjukkan model bisa mengalami fase sakit dalam jangka pendek.
- Periode 30 hari masih positif, tetapi jalannya kasar. Half-Kelly membuat profit lebih besar, tetapi drawdown 30 hari juga hampir -44%.
- Ini mendukung kesimpulan: edge ada, tetapi sizing tidak boleh terlalu percaya diri.

## 6. Monte Carlo: Skenario Ulang Masa Depan

Simulasi Monte Carlo memakai block bootstrap dari return 100 hari OOT. Artinya, kita mengacak ulang blok-blok return historis untuk melihat kemungkinan jalur masa depan.

Hasil 10.000 simulasi:

| Horizon | Versi | Median Return | P5 Return | Prob Akhir Rugi | Median MaxDD | Prob MaxDD <= -30% | Prob MaxDD <= -50% |
|---:|---|---:|---:|---:|---:|---:|---:|
| 100D | Baseline | +102.05% | -28.98% | 13.25% | -33.36% | 61.98% | 11.56% |
| 100D | Half-Kelly | +213.84% | -53.47% | 16.80% | -55.16% | 97.19% | 62.63% |
| 252D | Baseline | +493.50% | +17.11% | 3.43% | -44.58% | 92.52% | 34.33% |
| 252D | Half-Kelly | +1,728.07% | -8.96% | 5.61% | -69.72% | 99.99% | 93.61% |

Interpretasi:

- Baseline punya probabilitas rugi akhir tahun lebih rendah, tetapi hampir pasti tetap mengalami drawdown besar.
- Half-Kelly memberi upside ekstrem, tetapi hampir semua simulasi mengalami drawdown lebih dari -30%, dan 93.61% simulasi 252D mengalami drawdown lebih dari -50%.
- Untuk trader ritel, ini berarti half-Kelly mungkin terlalu agresif kecuali modal benar-benar dingin dan eksekusi leverage sangat disiplin.

Artifact:

- `kelly_baseline_vs_half_monte_carlo.csv`
- `monte_carlo/monte_equity_fan_100d.png`
- `monte_carlo/monte_equity_fan_252d.png`
- `monte_carlo/monte_maxdd_hist_100d.png`
- `monte_carlo/monte_maxdd_hist_252d.png`
- `monte_carlo/monte_return_cdf_100d.png`
- `monte_carlo/monte_return_cdf_252d.png`

## 7. Rekomendasi Praktis

### Jika tujuan utama adalah eksekusi realistis

Gunakan baseline atau baseline yang sedikit dinaikkan, bukan half-Kelly literal.

Aturan praktis:

- Cash-only konservatif: 25% per saham, maksimal 3 saham.
- Cash-only agresif: sekitar 33% per saham, maksimal 3 saham.
- Hindari leverage sebelum live pipeline dan slippage real benar-benar terbukti.

### Jika tetap ingin half-Kelly

Half-Kelly literal berarti:

- sekitar 47.2% per saham,
- sekitar 141.7% total gross jika 3 saham,
- butuh margin/leverage,
- bisa mengalami drawdown historis -58%.

Aturan wajib:

- Jangan pakai dana kebutuhan hidup.
- Jangan tambah manual di saham yang sama.
- Jangan override pick karena feeling.
- Jangan pakai jika tidak sanggup melihat penurunan modal 40-60%.
- Monitor data freshness sebelum entry. Jika `entry_price` atau `pre14` tidak siap, no trade.

## 8. Risiko Utama

### 8.1 Drawdown

Baseline drawdown -35.95% masih bisa dianggap wajar untuk strategi high-growth IDX second-liner. Tetapi half-Kelly drawdown -58.06% sudah masuk zona yang sulit secara psikologis.

### 8.2 Lookahead Bias

Audit sejauh ini tidak menemukan bukti lookahead fatal pada v25 T-1 quick. Namun risiko tetap harus dimonitor:

- broker features harus T-1,
- exit price tidak boleh jadi feature,
- cross-sectional feature harus tetap diaudit,
- live feature parity harus dijaga.

### 8.3 Inference Readiness

Backtest bagus tidak otomatis berarti sinyal live siap. Live inference harus punya:

- data T-1 lengkap,
- intraday hari T sampai jam 14/15,
- `entry_price` terisi,
- `pre14_*` terisi,
- model variant benar,
- Telegram heartbeat tidak blocked.

Pada 21 Mei 2026, bug `entry_price=0` sudah diperbaiki untuk jalur inference. Heartbeat juga sudah dibedakan antara `STANDBY` dan `STALE` supaya data hari berjalan sebelum jam 15 tidak salah dibaca sebagai blocker.

### 8.4 Artifact Freshness

PnL/report OOT v25 ini masih berbasis artifact sampai 6 Mei 2026. Untuk laporan yang benar-benar paling baru sampai 20/21 Mei, perlu refresh L1/L2 dan regenerate PnL artifact.

## 9. Keputusan Sementara

Baseline layak dipakai sebagai kandidat utama untuk eksekusi realistis, dengan catatan live data pipeline harus terus dipantau.

Half-Kelly layak dipakai sebagai stress-test dan upper-bound sizing, tetapi belum layak menjadi default untuk trader ritel karena:

- butuh leverage,
- drawdown sangat dalam,
- Monte Carlo menunjukkan risiko drawdown > -50% sangat tinggi,
- model masih perlu pembuktian live inference dan rolling refresh.

Rekomendasi saat ini:

| Tujuan | Pilihan |
|---|---|
| Live paper trading / small capital | Baseline |
| Cash-only agresif | Cap sekitar 33% per saham |
| Stress test / simulasi potensi | Half-Kelly |
| Default real-money leverage | Tidak direkomendasikan dulu |

## 10. Lampiran Artifact

File utama:

- `metrics.json`
- `feature_importance.csv`
- `walkforward_metrics.csv`
- `portfolio_daily.parquet`
- `valid_predictions.parquet`
- `kelly_baseline_vs_half_metrics.json`
- `kelly_baseline_vs_half_30d.csv`
- `kelly_baseline_vs_half_monte_carlo.csv`

Visual:

- `pnl_20d.png`
- `pnl_50d.png`
- `pnl_100d.png`
- `kelly_baseline_vs_half_equity_100d.png`
- `kelly_baseline_vs_half_drawdown_100d.png`
- `monte_carlo/monte_equity_fan_100d.png`
- `monte_carlo/monte_equity_fan_252d.png`
- `monte_carlo/monte_maxdd_hist_100d.png`
- `monte_carlo/monte_maxdd_hist_252d.png`
- `monte_carlo/monte_return_cdf_100d.png`
- `monte_carlo/monte_return_cdf_252d.png`

