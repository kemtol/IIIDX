# BSJP v10 Close10 Objective - Laporan Komprehensif Perbandingan

Tanggal: 23 April 2026
Pemilik: Stream riset BSJP (`idx/edges/bsjp_overnight_sl2`)

## Ringkasan Eksekutif
- Penyelarasan objective ke `close(T) -> close~10:00(T+1)` mengubah perilaku portofolio secara material.
- `v10_close10_obj` menghasilkan cumulative return jauh lebih tinggi (`+226.47%`) dibanding `v10_gapchar` (`+9.06%`) dan `v9d` (`+52.24%`).
- Risiko masih belum layak produksi: max drawdown tetap dalam (`-48.48%`), hanya sedikit lebih baik dari `v10_gapchar` (`-50.39%`) dan jauh lebih buruk dari `v9d` (`-29.11%`).
- Perbedaan utama antara `v10_gapchar` dan `v10_close10_obj` berasal dari perubahan target/objective, bukan dari ketidaksamaan parameter training.
- Tahap berikutnya harus memprioritaskan pengendalian tail-risk dan retuning policy/risk-overlay sebelum pertimbangan deploy.

## Ruang Lingkup dan Run yang Dibandingkan
| Run | Folder | Perubahan Utama | Jumlah Fitur | OOT Window |
|---|---|---|---:|---:|
| `v9d_classifier_ref` | `idx/model/BSJP/bsjp_v9d_classifier_ref` | Baseline referensi (objective open) | 223 | 100 hari |
| `v10_gapchar` | `idx/model/BSJP/bsjp_v10_gapchar` | Tambah keluarga fitur `gap_`, objective sama dengan baseline | 235 | 100 hari |
| `v10_close10_obj` | `idx/model/BSJP/bsjp_v10_close10_obj` | Fitur sama dengan `v10_gapchar`, objective diubah ke close10 | 235 | 100 hari |

## Perubahan yang Dilakukan
1. Menambahkan keluarga fitur karakter gap dengan prefix `gap_` pada generator datamart.
2. Regenerate datamart utama dengan pengecekan schema ketat.
3. Melatih `v10_gapchar` dengan konfigurasi training/policy/cost yang sama dengan keluarga baseline.
4. Membangun datamart objective-aligned untuk close10 dengan pemetaan bar deterministik.
5. Melatih `v10_close10_obj` dari datamart objective-aligned.

## Pemeriksaan Integritas Data dan Label
### Keamanan regenerate datamart
| Pemeriksaan | Hasil |
|---|---|
| Kolom lama yang hilang | 0 |
| Perubahan jumlah baris | `112,386 -> 112,386` |
| Perubahan jumlah kolom | `234 -> 246` |
| Kolom baru ditambahkan | 12 (`gap_...`) |

### Relabel objective (open objective vs close10 objective)
| Set Label | TP Rate | Mean Return | Return Std |
|---|---:|---:|---:|
| Datamart objective open | 37.42% | 0.108% | 8.9225% |
| Datamart objective close10 | 38.11% | 0.4255% | 9.6697% |

Interpretasi: relabel objective meningkatkan ekspektasi payoff sekaligus varians; ini secara natural menggeser ranking model dan profil risiko portofolio.

## Hasil Utama
### 1) Metrik klasifikasi OOT
| Metrik | v9d | v10_gapchar | v10_close10_obj |
|---|---:|---:|---:|
| AUC | 0.5948 | 0.5906 | 0.5772 |
| AUCPR | 0.4816 | 0.4775 | 0.4798 |
| Logloss | 0.6684 | 0.6698 | 0.6790 |
| Brier | 0.2377 | 0.2384 | 0.2430 |

### 2) Metrik portofolio OOT (métrik keputusan)
| Metrik | v9d | v10_gapchar | v10_close10_obj |
|---|---:|---:|---:|
| Cumulative net return | +52.24% | +9.06% | +226.47% |
| Max drawdown | -29.11% | -50.39% | -48.48% |
| Win-rate hari | 59% | 57% | 60% |
| Mean daily net return | 0.498% | 0.259% | 1.346% |
| Net expectancy/trade | 0.492% | 0.267% | 1.339% |
| Volatilitas harian | 4.12% | 6.33% | 5.74% |

### 2b) Snapshot stabilitas walk-forward
| Metrik | v9d | v10_gapchar | v10_close10_obj |
|---|---:|---:|---:|
| WF AUC mean | 0.6094 | 0.6064 | 0.5957 |
| WF AUC std | 0.0215 | 0.0257 | 0.0226 |
| WF AUCPR mean | 0.5117 | 0.5066 | 0.5249 |
| WF AUCPR std | 0.0214 | 0.0285 | 0.0193 |

### 3) Profil tail-risk (return net harian portofolio)
| Metrik Tail | v9d | v10_gapchar | v10_close10_obj |
|---|---:|---:|---:|
| Hari <= -3% | 6 | 13 | 10 |
| Hari <= -5% | 3 | 7 | 7 |
| Hari <= -8% | 1 | 5 | 3 |
| Hari <= -10% | 1 | 4 | 3 |
| Hari terburuk | -13.26% | -18.00% | -15.39% |

### 4) Profil payoff trade terpilih Top-3
| Metrik | v9d | v10_gapchar | v10_close10_obj |
|---|---:|---:|---:|
| Mean return trade terpilih | 0.892% | 0.665% | 1.739% |
| P90 | 3.61% | 5.36% | 10.48% |
| P95 | 5.86% | 8.14% | 13.96% |
| P99 | 19.61% | 19.66% | 30.51% |
| Jumlah > +20% | 2 | 3 | 7 |
| Jumlah < -20% | 0 | 5 | 4 |

### 5) Overlap pick (v10_gapchar vs v10_close10_obj)
- Overlap Top-3: `53 / 300` trade (`17.67%`).
- Kesimpulan: perubahan objective menghasilkan set pick harian yang berbeda besar; ini bukan perubahan kecil.

### 6) Pergeseran atribusi fitur
| Metrik Atribusi | v10_gapchar | v10_close10_obj |
|---|---:|---:|
| Total gain share fitur `gap_` | 4.10% | 11.44% |
| Fitur `gap_` non-zero | 8 / 12 | 9 / 12 |
| Fitur top #3 | `overnight_positive_rate60` | `vix_5d_avg` |
| Fitur top #4 | `overnight_positive_rate20` | `gap_up2_followthrough_freq_20d` |

Interpretasi: setelah objective diselaraskan ke close10, fitur karakter gap menjadi lebih berpengaruh dalam keputusan model.

## Diagnosis Mendalam: Kenapa v10_gapchar vs v10_close10_obj Berbeda
1. Hyperparameter sama, tetapi distribusi target berbeda.
2. Target close10 memiliki rata-rata payoff lebih tinggi dengan varians lebih lebar.
3. Frontier ranking bergeser ke kandidat ber-payoff tinggi, sehingga right-tail capture meningkat.
4. Left-tail tetap ada, sehingga drawdown masih dalam.
5. Metrik klasifikasi tidak membaik, tetapi payoff portofolio naik karena perubahan shape payoff pada top-k terpilih.

## Capaian yang Sudah Diperoleh
1. Menambahkan keluarga fitur `gap_` tanpa merusak schema lama.
2. Membentuk pipeline transform objective yang reproducible (`open -> close10`).
3. Memvalidasi bahwa objective alignment mampu mengangkat upside capture secara material.
4. Menghasilkan artifact lengkap untuk dua varian v10 (`metrics`, `predictions`, `portfolio`, `importance`).

## Gap / Risiko Saat Ini
1. Drawdown masih di luar batas yang layak untuk produksi.
2. Frekuensi tail-loss masih tinggi dibanding baseline referensi.
3. Kualitas sinyal masih cenderung payoff-driven, belum risk-adjusted dengan baik.
4. Policy warisan objective lama kemungkinan belum optimal untuk objective close10.

## Area of Growth (Roadmap Prioritas)
| Prioritas | Area | Kenapa Penting | Aksi | Success Gate |
|---|---|---|---|---|
| P0 | Kontrol tail-risk | MaxDD masih ekstrem | Tambah pre-trade veto dengan `gap_down2_freq`, regime flags, dan universe guard lebih ketat | MaxDD membaik material sambil menjaga expectancy positif |
| P0 | Retune policy untuk objective close10 | Policy masih copy objective lama | Re-grid `adaptive_threshold_quantile`, `max_weight_per_name`, top-k exposure | Tail-day (`<= -5%`, `<= -8%`) turun dengan return stabil |
| P1 | Ablation fitur `gap_` | Hindari fitur noisy/tidak berguna | Keep hanya fitur dengan incremental gain di OOT portfolio | OOT PnL/DD lebih baik dari full feature set |
| P1 | Regime-aware exposure scaling | Cegah loss berklaster saat regime buruk | Exposure dinamis saat stress regime | Klaster drawdown berkurang |
| P2 | Stabilitas multi-seed | Pastikan robust, bukan kebetulan | Train multi-seed dan buat leaderboard risk-adjusted | Dispersi hasil antar-seed rendah |

## Eksperimen Lanjutan yang Direkomendasikan (Immediate)
1. Gunakan `v10_close10_obj` sebagai base candidate.
2. Jalankan grid policy/risk overlay dengan bobot model tetap.
3. Pilih kandidat terbaik berbasis risk-adjusted criteria, bukan raw cumulative return; scoring wajib mencakup cumulative return, max drawdown, tail-day counts (`<= -5%`, `<= -8%`), dan konsistensi antar fold/subperiode.

## Artifact
- Baseline referensi: `idx/model/BSJP/bsjp_v9d_classifier_ref`
- Run v10 gap features: `idx/model/BSJP/bsjp_v10_gapchar`
- Run v10 objective close10: `idx/model/BSJP/bsjp_v10_close10_obj`
- Script transform objective: `idx/edges/bsjp_overnight_sl2/scripts/build_close10_objective_datamart.py`
- Datamart objective close10: `idx/data/Level_2_Datamart/training_datamart_bsjp_close10_v10.parquet`
