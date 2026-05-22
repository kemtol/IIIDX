# Laporan Analisis Strategi BSJP v25

**Evaluasi Sizing Baseline vs Half-Kelly**  
Tanggal laporan: **21 Mei 2026**  
Model: `v25_clean_t1quick_nl31_md100_l2.0_market_k3_w25`  
Objective: **close10** - beli menjelang penutupan hari T, jual sekitar sesi pembukaan 10:00 hari T+1  
Audience: trader ritel dan evaluator internal strategi BSJP

---

## 1. Ringkasan Eksekutif

Laporan ini mengevaluasi model BSJP v25 dari sudut pandang **position sizing**, bukan membandingkan dua model yang berbeda. Dua skenario yang diuji adalah:

1. **Baseline**: memakai policy asli model, maksimal 3 saham dengan bobot maksimum 25% per saham.
2. **Half-Kelly**: menaikkan exposure baseline sebesar `1.8888x`, berdasarkan estimasi Kelly dari distribusi return harian portofolio OOT.

Hasil pengujian pada **100 trading days OOT** menunjukkan bahwa baseline sudah menghasilkan pertumbuhan modal yang kuat, sementara half-Kelly meningkatkan potensi return secara signifikan tetapi memperbesar drawdown ke level yang jauh lebih berat secara psikologis dan operasional.

| Skenario | Exposure | Return 100D OOT | Max Drawdown | Rp10 juta menjadi | Catatan Utama |
|---|---:|---:|---:|---:|---|
| Baseline | 1.00x | +95.59% | -35.95% | Rp19.56 juta | Masih realistis untuk cash-only |
| Half-Kelly | 1.89x | +197.59% | -58.06% | Rp29.76 juta | Return tinggi, tetapi membutuhkan leverage dan toleransi DD sangat besar |

**Kesimpulan utama:** baseline lebih layak menjadi acuan eksekusi realistis. Half-Kelly berguna sebagai stress test dan batas atas agresivitas sizing, tetapi belum layak menjadi default real-money untuk trader ritel karena implied gross exposure mencapai sekitar **141.7%**.

---

## 2. Konteks Strategi

BSJP adalah strategi **Beli Sore Jual Pagi**. Model memilih saham menjelang penutupan pasar, lalu posisi dijual pada sesi pembukaan hari bursa berikutnya.

| Komponen | Deskripsi |
|---|---|
| Universe | Saham IDX, terutama saham aktif yang lolos filter harga dan biaya |
| Entry | Close sekitar jam 15 pada hari T |
| Exit | Open sekitar jam 10 pada hari T+1 |
| Maksimal posisi | 3 saham |
| Bobot baseline | Maksimum 25% per saham |
| Biaya asumsi | Roundtrip sekitar 0.40% |
| OOT window | 26 November 2025 - 6 Mei 2026 |
| Jumlah hari OOT | 100 trading days |

Model v25 ini bukan model pemburu ARA murni. Feature importance menunjukkan bahwa sinyal dominan berasal dari struktur intraday sebelum keputusan entry, terutama biaya/spread relatif, range harga, VWAP, dan turnover sesi siang.

| Rank | Feature | Interpretasi Operasional |
|---:|---|---|
| 1 | `pre14_tick_pct` | Estimasi friksi harga relatif terhadap harga saham |
| 2 | `pre14_range_pct` | Aktivitas dan volatilitas harga sebelum sore |
| 3 | `pre14_vwap` | Harga rata-rata berbobot volume sebelum entry |
| 4 | `pre14_spread_cost_est` | Estimasi biaya spread masuk-keluar |
| 5 | `pre14_pm_turnover` | Nilai transaksi sesi siang |

---

## 3. Definisi Sizing

### 3.1 Baseline

Baseline adalah policy asli model:

- maksimal 3 saham per hari,
- bobot maksimum 25% per saham,
- total gross exposure maksimum sekitar 75%,
- tidak membutuhkan leverage jika modal tersedia dalam bentuk cash.

### 3.2 Half-Kelly

Half-Kelly dalam laporan ini dihitung dari **return harian portofolio baseline**, bukan dari probability individual saham. Rumus yang dipakai adalah pendekatan empirical Kelly, yaitu mencari multiplier yang memaksimalkan rata-rata log return:

```text
maximize average(log(1 + f * daily_return))
```

Hasil estimasi:

| Metrik Kelly | Nilai |
|---|---:|
| Approx full Kelly (`mean / variance`) | 3.5678x |
| Empirical full Kelly | 3.7776x |
| Half-Kelly | 1.8888x |
| Worst-day ruin limit historis | 9.4990x |

Implikasi sizing:

| Skenario | Max weight per saham | Gross exposure jika 3 saham |
|---|---:|---:|
| Baseline | 25.00% | 75.00% |
| Half-Kelly literal | 47.22% | 141.66% |

Dengan demikian, half-Kelly literal **bukan cash-only**. Jika strategi dibatasi cash-only, scaling yang lebih realistis adalah sekitar `1.333x` dari baseline, karena `75% x 1.333 ~= 100%`.

---

## 4. Hasil Historis OOT: Baseline vs Half-Kelly

### 4.1 Equity Curve

Grafik berikut menunjukkan pertumbuhan modal pada 100 trading days OOT. Baseline menghasilkan pertumbuhan yang kuat dengan drawdown besar tetapi masih dalam batas yang relatif dapat diterima untuk strategi high-growth. Half-Kelly menghasilkan pertumbuhan lebih tinggi, tetapi jalurnya jauh lebih volatile.

![Baseline vs Half-Kelly Equity Curve](./kelly_baseline_vs_half_equity_100d.png)

| Metrik | Baseline | Half-Kelly |
|---|---:|---:|
| Cumulative return | +95.59% | +197.59% |
| Final capital dari Rp10 juta | Rp19.56 juta | Rp29.76 juta |
| Annualized return | +442.23% | +1,461.45% |
| Sharpe annualized | 2.65 | 2.65 |
| Win days | 50 / 100 | 50 / 100 |
| Best day | +13.61% | +25.71% |
| Worst day | -10.53% | -19.88% |

Sharpe ratio sama karena half-Kelly hanya menskalakan return baseline. Return dan volatilitas naik proporsional. Karena itu, Sharpe tidak boleh dibaca sebagai bukti bahwa half-Kelly lebih aman.

### 4.2 Drawdown

![Baseline vs Half-Kelly Drawdown](./kelly_baseline_vs_half_drawdown_100d.png)

| Metrik Risiko | Baseline | Half-Kelly |
|---|---:|---:|
| Max drawdown | -35.95% | -58.06% |
| Ulcer Index | 15.06% | 26.22% |
| Calmar Ratio | 12.30 | 25.17 |

Baseline mengalami drawdown maksimum sekitar -36%. Untuk strategi IDX second-liner dengan target return tinggi, level ini masih dapat dipertimbangkan. Half-Kelly mengalami drawdown maksimum sekitar -58%, yang secara praktis hanya cocok untuk modal yang benar-benar dingin dan trader dengan toleransi tekanan sangat tinggi.

---

## 5. Performa Rolling Window

Analisis rolling window membantu melihat pengalaman investor jika mulai masuk di fase yang berbeda. Periode terakhir OOT menunjukkan bahwa model sempat mengalami fase pendek yang buruk, terutama pada window 14 hari terakhir.

| Window Terakhir | Baseline Return | Baseline MaxDD | Half-Kelly Return | Half-Kelly MaxDD |
|---:|---:|---:|---:|---:|
| 7 trading days | -1.88% | -4.19% | -3.72% | -7.84% |
| 14 trading days | -22.13% | -21.54% | -38.57% | -37.68% |
| 30 trading days | +22.31% | -25.64% | +40.86% | -43.77% |
| 90 trading days | +49.78% | -35.95% | +83.37% | -58.06% |
| 100 trading days | +95.59% | -35.95% | +197.59% | -58.06% |

Interpretasi:

- Window 14 hari terakhir adalah fase stress yang penting. Baseline turun -22.13%, half-Kelly turun -38.57%.
- Window 30 hari tetap positif, tetapi perjalanan tidak mulus.
- Half-Kelly meningkatkan return saat model bekerja, tetapi memperbesar kerugian secara tajam saat model masuk fase buruk.

Data detail window 30 hari tersedia di:

```text
kelly_baseline_vs_half_30d.csv
```

---

## 6. Monte Carlo dan Stress Test

Monte Carlo dilakukan dengan block bootstrap dari return harian OOT. Simulasi ini tidak memprediksi masa depan secara presisi, tetapi memberi gambaran distribusi kemungkinan jika pola return historis berulang dalam urutan yang berbeda.

### 6.1 Ringkasan Monte Carlo Baseline vs Half-Kelly

| Horizon | Skenario | Median Return | P5 Return | Probabilitas Akhir Rugi | Median MaxDD | Prob. MaxDD <= -30% | Prob. MaxDD <= -50% |
|---:|---|---:|---:|---:|---:|---:|---:|
| 100D | Baseline | +102.05% | -28.98% | 13.25% | -33.36% | 61.98% | 11.56% |
| 100D | Half-Kelly | +213.84% | -53.47% | 16.80% | -55.16% | 97.19% | 62.63% |
| 252D | Baseline | +493.50% | +17.11% | 3.43% | -44.58% | 92.52% | 34.33% |
| 252D | Half-Kelly | +1,728.07% | -8.96% | 5.61% | -69.72% | 99.99% | 93.61% |

Kesimpulan Monte Carlo:

- Baseline masih memiliki risiko drawdown besar, tetapi probabilitas akhir rugi lebih terkendali.
- Half-Kelly memberi upside yang ekstrem, namun hampir seluruh simulasi mengalami drawdown lebih dari -30%.
- Pada horizon 252 hari, half-Kelly memiliki probabilitas drawdown lebih dari -50% sebesar 93.61%. Ini menjadikannya tidak cocok sebagai default untuk trader ritel.

### 6.2 Visual Monte Carlo

**Fan Chart 100 Hari**

![Monte Carlo Equity Fan 100D](./monte_carlo/monte_equity_fan_100d.png)

**Fan Chart 252 Hari**

![Monte Carlo Equity Fan 252D](./monte_carlo/monte_equity_fan_252d.png)

**Return Distribution 100 Hari**

![Monte Carlo Return CDF 100D](./monte_carlo/monte_return_cdf_100d.png)

**Return Distribution 252 Hari**

![Monte Carlo Return CDF 252D](./monte_carlo/monte_return_cdf_252d.png)

**Max Drawdown Histogram 100 Hari**

![Monte Carlo MaxDD Histogram 100D](./monte_carlo/monte_maxdd_hist_100d.png)

**Max Drawdown Histogram 252 Hari**

![Monte Carlo MaxDD Histogram 252D](./monte_carlo/monte_maxdd_hist_252d.png)

---

## 7. Visual PnL Baseline

Grafik berikut adalah artifact PnL baseline yang dihasilkan saat training. Grafik ini membantu membaca kondisi performa model pada horizon pendek, menengah, dan penuh OOT.

**PnL 20 Trading Days**

![PnL 20D](./pnl_20d.png)

**PnL 50 Trading Days**

![PnL 50D](./pnl_50d.png)

**PnL 100 Trading Days**

![PnL 100D](./pnl_100d.png)

---

## 8. Penilaian Risiko

### 8.1 Risiko Drawdown

Baseline memiliki MaxDD -35.95%. Dalam konteks strategi agresif yang menargetkan saham IDX dengan pergerakan cepat, drawdown ini masih dapat dipertimbangkan. Namun, investor harus menerima bahwa fase rugi besar tetap mungkin terjadi.

Half-Kelly memiliki MaxDD -58.06%. Ini bukan sekadar volatilitas biasa. Pada level ini, banyak trader akan berhenti mengikuti sistem sebelum strategi sempat pulih.

### 8.2 Risiko Leverage

Half-Kelly literal membutuhkan gross exposure sekitar 141.66%. Artinya, strategi harus menggunakan margin atau leverage. Risiko tambahan yang tidak tercermin penuh di backtest:

- biaya margin,
- forced liquidation,
- gagal eksekusi di saham tidak likuid,
- slippage lebih besar saat ukuran order naik,
- disiplin eksekusi yang lebih sulit.

### 8.3 Risiko Lookahead

Audit saat ini tidak menunjukkan bukti lookahead fatal pada model v25 T-1 quick. Namun, risiko ini tetap menjadi area kontrol utama. Aturan yang harus terus dijaga:

- broker features harus T-1,
- exit price tidak boleh jadi feature,
- data intraday harus tersedia sebelum jam keputusan,
- feature live harus konsisten dengan feature training.

### 8.4 Risiko Operasional Inference

Backtest yang baik tidak otomatis menjamin sinyal live siap dikirim. Untuk live inference, sistem harus memenuhi syarat berikut:

- data T-1 lengkap,
- data intraday hari T tersedia sampai minimal jam 14/15,
- `entry_price` terisi,
- `pre14_*` terisi,
- model variant sesuai,
- heartbeat tidak berada dalam status blocked.

Pada 21 Mei 2026, bug `entry_price=0` pada jalur inference sudah diperbaiki. Heartbeat juga telah dipisahkan antara kondisi `STANDBY` dan `STALE`, sehingga data hari berjalan sebelum jam 15 tidak lagi salah dibaca sebagai kegagalan.

---

## 9. Rekomendasi Sizing

| Profil Pengguna | Rekomendasi |
|---|---|
| Trader ritel konservatif | Gunakan baseline atau lebih kecil |
| Trader ritel agresif cash-only | Pertimbangkan cap sekitar 33% per saham, total gross mendekati 100% |
| Trader profesional dengan margin | Half-Kelly hanya sebagai batas atas eksperimen, bukan default |
| Paper trading / live validation | Baseline lebih tepat untuk validasi awal |

Rekomendasi utama:

1. Jadikan **baseline** sebagai acuan eksekusi utama.
2. Jangan menjadikan half-Kelly sebagai default real-money sampai ada bukti live yang lebih panjang.
3. Jika ingin meningkatkan sizing, lakukan bertahap: `1.0x -> 1.25x -> 1.33x`, bukan langsung `1.89x`.
4. Ukur ulang Kelly setelah ada live result yang cukup, bukan hanya dari OOT historis.

---

## 10. Keputusan Sementara

Model v25 baseline dapat dipertahankan sebagai kandidat strategi BSJP yang agresif tetapi masih realistis. Half-Kelly menunjukkan potensi return yang jauh lebih besar, namun profil risikonya terlalu ekstrem untuk dijadikan default.

Keputusan sementara:

| Area | Status |
|---|---|
| Baseline research result | Layak dipertahankan |
| Half-Kelly as default sizing | Tidak direkomendasikan |
| Half-Kelly as stress test | Berguna |
| Live inference | Perlu monitoring harian, bug entry price sudah diperbaiki |
| Report/PnL recency | Perlu refresh L1/L2 untuk memperpanjang artifact setelah 6 Mei 2026 |

---

## 11. Artifact Register

### Data dan Metrik

| File | Keterangan |
|---|---|
| `metrics.json` | Metrik training, OOT, policy, dan metadata model |
| `feature_importance.csv` | Feature importance LightGBM |
| `walkforward_metrics.csv` | Ringkasan walk-forward validation |
| `portfolio_daily.parquet` | Return harian portofolio baseline |
| `valid_predictions.parquet` | Prediksi valid/OOT per saham |
| `kelly_baseline_vs_half_metrics.json` | Metrik sizing baseline dan half-Kelly |
| `kelly_baseline_vs_half_30d.csv` | Detail window 30 hari terakhir |
| `kelly_baseline_vs_half_monte_carlo.csv` | Monte Carlo baseline vs half-Kelly |

### Visual

| File | Keterangan |
|---|---|
| `kelly_baseline_vs_half_equity_100d.png` | Equity curve baseline vs half-Kelly |
| `kelly_baseline_vs_half_drawdown_100d.png` | Drawdown baseline vs half-Kelly |
| `pnl_20d.png` | PnL baseline 20 hari |
| `pnl_50d.png` | PnL baseline 50 hari |
| `pnl_100d.png` | PnL baseline 100 hari |
| `monte_carlo/monte_equity_fan_100d.png` | Monte Carlo fan chart 100 hari |
| `monte_carlo/monte_equity_fan_252d.png` | Monte Carlo fan chart 252 hari |
| `monte_carlo/monte_return_cdf_100d.png` | Distribusi return Monte Carlo 100 hari |
| `monte_carlo/monte_return_cdf_252d.png` | Distribusi return Monte Carlo 252 hari |
| `monte_carlo/monte_maxdd_hist_100d.png` | Histogram MaxDD Monte Carlo 100 hari |
| `monte_carlo/monte_maxdd_hist_252d.png` | Histogram MaxDD Monte Carlo 252 hari |
