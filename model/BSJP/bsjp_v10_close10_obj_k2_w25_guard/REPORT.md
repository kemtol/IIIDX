# Laporan Kuantitatif Institusional - BSJP v10 (Close10, K2-W25 Guard)

Tanggal: 23 April 2026  
Penyusun: Tim Quant Riset BSJP  
Status Dokumen: Siap Publikasi Internal ke Shareholder

## 1. Ringkasan Eksekutif
Laporan ini merangkum hasil riset berjenjang dari baseline `v9d` sampai kandidat risk-managed terbaru `v10_close10_obj_k2_w25_guard`.

Kesimpulan utama:
1. Penyelarasan objective ke horizon exit `10:00` meningkatkan potensi upside secara material.
2. Tanpa guard policy, objective baru menghasilkan drawdown yang terlalu dalam untuk standard institusional.
3. Policy guard `top_k=2` dan `max_weight=0.25` menurunkan tail-risk secara signifikan tanpa perlu generate fitur/data baru.
4. Kandidat paling seimbang saat ini adalah `bsjp_v10_close10_obj_k2_w25_guard`.

## 2. Tujuan, Cakupan, dan Standard Evaluasi
Tujuan riset:
1. Meningkatkan return terukur pada OOT 100 hari.
2. Menurunkan drawdown dan frekuensi tail-loss agar lebih layak untuk rollout bertahap.
3. Menjaga prosedur validasi yang dapat diaudit (reproducible, no-lookahead).

Cakupan evaluasi:
1. Horizon OOT: 100 hari trading.
2. Data total: 112.386 baris, 728 hari trading.
3. Komparasi run:
- `v9d_classifier_ref`
- `v10_gapchar`
- `v10_close10_obj`
- `v10_close10_obj_k2_guard`
- `v10_close10_obj_k2_w25_guard`

## 3. Tata Kelola Data dan Integritas Metodologi
Kontrol integritas yang diterapkan:
1. Split kronologis walk-forward + OOT (tanpa random split).
2. Feature no-lookahead (fitur hanya memakai informasi sampai close hari T).
3. Cost model konsisten antar run:
- Buy cost 10 bps
- Sell cost 20 bps
- Slippage 5 bps per side
- Total roundtrip 40 bps

Validasi perubahan datamart:
1. Regenerate datamart tidak menghapus kolom lama (`missing_old_columns=0`).
2. Baris tetap (`112.386`).
3. Penambahan kolom hanya berasal dari keluarga `gap_` (12 kolom).

## 4. Narasi Perjalanan Eksperimen
### 4.1 Baseline (`v9d`)
Menjadi titik acuan performa dan risiko.

### 4.2 `v10_gapchar`
Menambah fitur karakter gap (`gap_`) dengan objective lama (open objective). Hasil: performa turun dan risiko memburuk.

### 4.3 `v10_close10_obj`
Objective diubah menjadi `close(T) -> close~10:00(T+1)` dengan fitur sama. Hasil: return melonjak, tetapi max drawdown masih terlalu tinggi.

### 4.4 Policy Overlay Tanpa Data Baru
Tanpa retrain fitur baru, dilakukan guard policy:
1. `k2_guard` (`top_k=2`, `max_weight=0.34`)
2. `k2_w25_guard` (`top_k=2`, `max_weight=0.25`)

Hasil: tail-risk turun signifikan, dengan return masih kuat.

## 5. Hasil Komparatif Utama
### 5.1 Kinerja Portofolio OOT
| Run | Cumulative Net Return | Max Drawdown | Win-Rate Day | Mean Daily Net | Daily Vol |
|---|---:|---:|---:|---:|---:|
| v9d_ref | +52.24% | -29.11% | 59% | 0.498% | 4.12% |
| v10_gapchar | +9.06% | -50.39% | 57% | 0.259% | 6.33% |
| v10_close10_obj | +226.47% | -48.48% | 60% | 1.346% | 5.74% |
| v10_close10_obj_k2_guard | +174.85% | -34.25% | 59% | 1.135% | 5.02% |
| v10_close10_obj_k2_w25_guard | +115.21% | -26.04% | 59% | 0.834% | 3.69% |

### 5.2 Statistik Risk-Adjusted (Lengkap Seluruh Run)
Asumsi perhitungan:
1. Sharpe/Sortino diannualisasi dengan akar 252 hari trading.
2. Risk-free rate diasumsikan 0 untuk komparasi internal antar run.
3. VaR 95% = kuantil 5% return harian, ES 95% = rata-rata return di bawah/tepat VaR.
4. Calmar = `(mean_daily_return * 252) / |max_drawdown|`.

| Run | Sharpe (ann.) | Sortino (ann.) | Calmar | VaR 95% harian | ES 95% harian |
|---|---:|---:|---:|---:|---:|
| v9d_ref | 1.91 | 3.35 | 4.31 | -3.57% | -6.66% |
| v10_gapchar | 0.65 | 1.10 | 1.29 | -5.40% | -12.18% |
| v10_close10_obj | 3.70 | 6.18 | 7.00 | -5.37% | -10.54% |
| v10_close10_obj_k2_guard | 3.57 | 5.74 | 8.35 | -4.32% | -9.62% |
| v10_close10_obj_k2_w25_guard | 3.57 | 5.74 | 8.07 | -3.17% | -7.07% |

Interpretasi:
1. `v10_gapchar` memiliki profil risk-adjusted terlemah (Sharpe/Sortino rendah, ES paling dalam).
2. `v10_close10_obj` unggul pada return, tetapi tail-risk tetap tinggi.
3. `v10_close10_obj_k2_w25_guard` memberi profil downside terbaik (VaR/ES paling rendah) dengan Sharpe/Sortino tetap tinggi.

### 5.3 Metrik Risiko Tambahan (Omega dan Ulcer Index)
| Run | Omega (MAR=0) | Ulcer Index | Max Drawdown |
|---|---:|---:|---:|
| v9d_ref | 1.66 | 15.61% | -29.11% |
| v10_gapchar | 1.20 | 27.71% | -50.39% |
| v10_close10_obj | 2.12 | 19.58% | -48.48% |
| v10_close10_obj_k2_guard | 2.15 | 14.94% | -34.25% |
| v10_close10_obj_k2_w25_guard | 2.15 | 11.12% | -26.04% |

Interpretasi:
1. `v10_gapchar` memiliki ulcer index tertinggi, mengonfirmasi fase underwater yang panjang dan dalam.
2. Overlay guard meningkatkan kualitas distribusi return (Omega naik) sekaligus memendekkan kedalaman drawdown (Ulcer turun).
3. `v10_close10_obj_k2_w25_guard` saat ini menjadi profil downside paling sehat di panel.

### 5.4 Tail-Day Diagnostics
| Run | Hari <= -5% | Hari <= -8% | Hari <= -10% | Worst Day |
|---|---:|---:|---:|---:|
| v10_close10_obj | 7 | 3 | 3 | -15.39% |
| v10_close10_obj_k2_guard | 5 | 3 | 3 | -13.47% |
| v10_close10_obj_k2_w25_guard | 3 | 3 | 0 | -9.90% |

### 5.5 Stabilitas Rolling 20 Hari (Kandidat Utama)
Ringkasan `v10_close10_obj_k2_w25_guard` berdasarkan rolling window 20 hari:
1. Snapshot terakhir (21 April 2026): rolling Sharpe `4.90`, rolling max drawdown `-4.18%`, rolling win-rate `60%`.
2. Distribusi rolling Sharpe (P10/P50/P90): `-3.55 / 5.04 / 7.93`.
3. Distribusi rolling max drawdown 20 hari (P10/P50/P90): `-24.04% / -5.53% / -3.61%`.
4. Distribusi rolling win-rate 20 hari (P10/P50/P90): `35% / 60% / 80%`.

Implikasi:
1. Model tetap memiliki fase lemah (terlihat di kuantil bawah Sharpe), sehingga exposure control tetap wajib.
2. Median rolling menunjukkan performa risk-adjusted kuat dan stabil pada mayoritas subperiode.
3. Guard policy menekan kedalaman drawdown jangka pendek ke koridor yang lebih operasional.

### 5.6 Monte Carlo Simulation (Block Bootstrap)
Konfigurasi simulasi:
1. Sumber return: `portfolio_daily.parquet` kandidat `v10_close10_obj_k2_w25_guard` (100 hari, 13 November 2025 sampai 21 April 2026).
2. Jumlah jalur simulasi: `10.000` path.
3. Metode sampling: `block bootstrap` dengan panjang blok `5` hari.
4. Horizon evaluasi: `100` hari dan `252` hari.
5. Seed simulasi: `20260423`.

Hasil utama simulasi:
| Horizon | Median Terminal Return | Prob. Terminal Return < 0 | Mean MaxDD | Prob. MaxDD <= -20% | Prob. MaxDD <= -25% | Prob. MaxDD <= -30% | P95 Worst MaxDD |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100 hari | +113.27% | 2.70% | -18.89% | 39.55% | 19.19% | 9.46% | -33.76% |
| 252 hari | +592.16% | 0.18% | -25.29% | 72.97% | 43.59% | 24.50% | -40.20% |

Interpretasi governance:
1. Distribusi return Monte Carlo tetap condong positif, tetapi distribusi drawdown menunjukkan tail-risk yang tidak kecil.
2. Pada horizon 252 hari, peluang mengalami drawdown lebih dalam dari `-25%` masih `43.59%`, sehingga sizing dan limit risiko wajib ketat.
3. Monte Carlo dipakai sebagai stress-distribution tool, bukan proyeksi titik hasil masa depan.
4. Asumsi stasioneritas return tetap menjadi keterbatasan utama; hasil harus dibaca bersama stress test operasional.

## 6. Diagnosa Penyebab Perbedaan Kinerja
### 6.1 Kenapa `v10_gapchar` dan `v10_close10_obj` bisa sangat berbeda
1. Hyperparameter hampir sama, tetapi target yang dipelajari berbeda.
2. Objective close10 memiliki distribusi payoff lebih tinggi sekaligus lebih volatil.
3. Ranking kandidat berubah besar (overlap top-3 rendah), sehingga hasil portofolio berubah drastis.

### 6.2 Kenapa guard policy efektif meski tanpa data/features baru
1. Pengurangan konsentrasi posisi (`max_weight`) langsung menurunkan dampak nama ekstrem.
2. Pengurangan jumlah posisi aktif (`top_k`) mengurangi eksposur tail event per hari.
3. Efek langsung terlihat pada VaR/ES dan penghapusan hari `<= -10%`.

## 7. Apa yang Sudah Dicapai
1. Pipeline objective-aligned berhasil dibangun dan tervalidasi.
2. Struktur data tetap konsisten dan dapat diaudit.
3. Ditemukan kandidat dengan kompromi return-risiko paling sehat saat ini.
4. Seluruh artefak sudah siap untuk due diligence lanjutan.

## 8. Area of Growth Terdekat (Tanpa Generate Feature/Data Baru)
Prioritas 30 hari ke depan:
1. Policy refinement lanjutan pada kandidat `k2_w25_guard`.
2. Kalibrasi threshold quantile.
3. Stress test cap bobot per nama.
4. Validasi subperiode drawdown.
5. Regime-aware exposure layer (berbasis sinyal makro yang sudah ada di fitur).
6. Risk committee gate untuk batas harian, weekly stop, dan capacity monitoring.
7. Multi-seed robustness check untuk memastikan hasil bukan kebetulan satu seed.

## 9. Rekomendasi Operasional ke Shareholder
Rekomendasi saat ini:
1. Jadikan `bsjp_v10_close10_obj_k2_w25_guard` sebagai kandidat utama untuk fase pilot bertahap.
2. Gunakan kontrol risiko operasional ketat pada fase pilot.
3. Tetapkan KPI evaluasi pilot berbasis risk-adjusted, bukan return absolut saja.

KPI minimum untuk lanjut scaling:
1. Max drawdown pilot tidak melampaui batas governance internal.
2. VaR/ES harian stabil pada koridor yang disetujui risk committee.
3. Konsistensi performa lintas subperiode tetap terjaga.

## 10. Catatan Risiko dan Disclaimer
1. Hasil ini berbasis backtest OOT, bukan jaminan hasil masa depan.
2. Risiko likuiditas, slippage aktual, dan event-driven gap tetap material.
3. Dokumen ini bersifat informasi internal shareholder dan bukan ajakan investasi publik.

## 11. Referensi Artefak
Kandidat utama:
- [metrics.json](/home-ssd/mkemalw/Projects/MMMACHINE/idx/model/BSJP/bsjp_v10_close10_obj_k2_w25_guard/metrics.json)
- [portfolio_daily.parquet](/home-ssd/mkemalw/Projects/MMMACHINE/idx/model/BSJP/bsjp_v10_close10_obj_k2_w25_guard/portfolio_daily.parquet)
- [valid_predictions.parquet](/home-ssd/mkemalw/Projects/MMMACHINE/idx/model/BSJP/bsjp_v10_close10_obj_k2_w25_guard/valid_predictions.parquet)
- [appendix_risk_metrics.csv](/home-ssd/mkemalw/Projects/MMMACHINE/idx/model/BSJP/bsjp_v10_close10_obj_k2_w25_guard/appendix_risk_metrics.csv)
- [appendix_rolling20_candidate.csv](/home-ssd/mkemalw/Projects/MMMACHINE/idx/model/BSJP/bsjp_v10_close10_obj_k2_w25_guard/appendix_rolling20_candidate.csv)
- [appendix_rolling20_panel.csv](/home-ssd/mkemalw/Projects/MMMACHINE/idx/model/BSJP/bsjp_v10_close10_obj_k2_w25_guard/appendix_rolling20_panel.csv)
- [monte_summary_metrics.csv](/home-ssd/mkemalw/Projects/MMMACHINE/idx/model/BSJP/bsjp_v10_close10_obj_k2_w25_guard/monte_summary_metrics.csv)
- [monte_config.json](/home-ssd/mkemalw/Projects/MMMACHINE/idx/model/BSJP/bsjp_v10_close10_obj_k2_w25_guard/monte_config.json)
- [monte_equity_fan.png](/home-ssd/mkemalw/Projects/MMMACHINE/idx/model/BSJP/bsjp_v10_close10_obj_k2_w25_guard/monte_equity_fan.png)
- [monte_maxdd_hist.png](/home-ssd/mkemalw/Projects/MMMACHINE/idx/model/BSJP/bsjp_v10_close10_obj_k2_w25_guard/monte_maxdd_hist.png)
- [monte_return_cdf.png](/home-ssd/mkemalw/Projects/MMMACHINE/idx/model/BSJP/bsjp_v10_close10_obj_k2_w25_guard/monte_return_cdf.png)

Run pembanding:
- [v9d metrics](/home-ssd/mkemalw/Projects/MMMACHINE/idx/model/BSJP/bsjp_v9d_classifier_ref/metrics.json)
- [v10_gapchar metrics](/home-ssd/mkemalw/Projects/MMMACHINE/idx/model/BSJP/bsjp_v10_gapchar/metrics.json)
- [v10_close10_obj metrics](/home-ssd/mkemalw/Projects/MMMACHINE/idx/model/BSJP/bsjp_v10_close10_obj/metrics.json)
- [v10_close10_obj_k2_guard metrics](/home-ssd/mkemalw/Projects/MMMACHINE/idx/model/BSJP/bsjp_v10_close10_obj_k2_guard/metrics.json)

Lampiran proses objective alignment:
- [build_close10_objective_datamart.py](/home-ssd/mkemalw/Projects/MMMACHINE/idx/edges/bsjp_overnight_sl2/scripts/build_close10_objective_datamart.py)
- [training_datamart_bsjp_close10_v10.parquet](/home-ssd/mkemalw/Projects/MMMACHINE/idx/data/Level_2_Datamart/training_datamart_bsjp_close10_v10.parquet)
