# bsjp_v12 — Planning & Execution

**Status**: PLANNED  
**Tanggal**: 2026-04-24  

---

## Tujuan

Virgin test params grid winner (`md100_lam1_0_gain0_1`) pada **overnight datamart** (BSJP asli).

Grid search sebelumnya dijalankan di close10 datamart — ini adalah pertama kalinya params ini dicoba di overnight. Tujuannya untuk mengetahui apakah sweet spot regularisasi yang ditemukan di close10 juga berlaku untuk overnight.

---

## Konteks & Hipotesis

| Model referensi | Params | Datamart | CumRet OOT | Overfit gap |
|----------------|--------|----------|-----------|-------------|
| `bsjp_v7` | md=500, λ=2.0 (ketat) | overnight | +102.8% | 0.031 ✅ |
| `bsjp_v11_hyperopt` | md=50, λ=0.5 (longgar) | overnight | +1667.7% | 0.065 ❌ |
| `grid_fine/md100_lam1_0_gain0_1` | md=100, λ=1.0 (medium) | **close10** | +3256.5% | 0.073 ❌ |

**Hipotesis**: params medium (md=100, λ=1.0) di overnight akan menghasilkan sweet spot antara v7 dan v11 — return lebih tinggi dari v7, overfit gap lebih sehat dari v11.

---

## Keputusan Desain

| Aspek | Nilai | Alasan |
|-------|-------|--------|
| Datamart | `training_datamart_bsjp_overnight.parquet` | BSJP = overnight, default script |
| `min_data_in_leaf` | **100** | Grid winner — medium antara 500 (v7) dan 50 (v11) |
| `lambda_l1` | **1.0** | Grid winner |
| `lambda_l2` | **1.0** | Grid winner |
| `min_gain_to_split` | **0.1** | Grid winner — sama dengan v7, lebih konservatif dari v11 (0.01) |
| `num_leaves` | **63** | Grid winner — lebih ekspresif dari v7 (31) |
| `max_depth` | **6** | Grid winner — lebih dalam dari v7 (4) |
| `feature_prune_top_n` | **0** | Pakai semua fitur, tidak prune |
| `oot_valid_days` | **100** | Standar semua run sebelumnya |
| `tp_pct` | **0.01** | Konsisten dengan v7, v10, v11 |
| `sl_pct` | **-0.02** | Konsisten dengan v7, v10, v11 |
| Policy | **k=3, rank_weights=0.6,0.3,0.1** | Aggressive — virgin test, lihat raw model performance dulu |

> Policy k=3 dipilih untuk virgin test agar hasil model tidak tertutupi efek sizing. Jika v12 menjanjikan, jalankan varian `bsjp_v12_guard` dengan k=2, max_weight=0.25 untuk compare dengan kandidat institusional.

---

## Command

Jalankan dari `idx/` directory:

```bash
source ../.venv/bin/activate
cd edges/bsjp_overnight_sl2/scripts

python train_lightgbm.py \
  --output-dir ../../../model/BSJP/bsjp_v12 \
  --min-data-in-leaf 100 \
  --lambda-l1 1.0 \
  --lambda-l2 1.0 \
  --min-gain-to-split 0.1 \
  --num-leaves 63 \
  --max-depth 6 \
  --oot-valid-days 100 \
  --feature-prune-top-n 0 \
  --tp-pct 0.01 \
  --sl-pct -0.02 \
  --conviction-top-k 3 \
  --rank-weights "0.6,0.3,0.1"
```

---

## Kriteria Evaluasi

Setelah run selesai, cek metrics.json dengan standar berikut:

| Metrik | Target | Referensi |
|--------|--------|-----------|
| Overfit gap | **< 0.050** | v11 gagal di 0.065, v7 sehat di 0.031 |
| AUC OOT | > 0.590 | v7=0.602, v11=0.595 |
| CumRet OOT | > +102.8% | Harus beat v7 untuk justified |
| MaxDD | < -45% | v7=-30.2%, toleransi lebih tinggi untuk aggressive policy |
| Vol/hari | < 10% | v7=3.79%, v11=14.94% — target di tengah |
| Fold AUC stability | Tidak ada fold < 0.560 | Indikator kestabilan antar periode |

---

## Artefak yang Dihasilkan

### Auto-generated oleh `train_lightgbm.py`

| File | Isi |
|------|-----|
| `metrics.json` | Semua metrik lengkap: AUC, portfolio OOT, overfit gap, params, execution config |
| `model_lightgbm_opening_tp3.txt` | Model LightGBM yang sudah ditraining (bisa di-load ulang untuk inference) |
| `feature_importance.csv` | Ranking feature importance berdasarkan gain |
| `valid_predictions.parquet` | Prediksi probabilitas per saham per hari di OOT window |
| `portfolio_daily.parquet` | Simulasi portofolio harian di OOT (modal, return, posisi per hari) |
| `walkforward_metrics.csv` | AUC dan metrics per fold walk-forward |

### Post-training (manual / script terpisah)

| File | Isi | Prioritas |
|------|-----|-----------|
| `REPORT.md` | Laporan kuantitatif institusional | Setelah v12_guard |
| `monte_*.png/csv/json` | Monte Carlo 10k paths (equity fan, maxdd hist, return CDF) | Wajib sebelum produksi |
| `pick_day_by_day_100d.csv` | Detail picks harian selama OOT 100 hari | Opsional |
| `pnl_growth_chart_*.png` | Grafik equity curve | Opsional |
| `kelly_*.csv/png` | Analisis Kelly criterion untuk sizing optimal | Opsional |

> Untuk v12 virgin test, yang penting dulu hanya auto-generated (6 file). Monte Carlo dijalankan setelah ada varian guard.

---

## Next Steps Setelah Run

1. **Jika overfit gap < 0.050 dan return > v7**: lanjut ke `bsjp_v12_guard` (k=2, max_weight=0.25) untuk bandingkan dengan kandidat institusional
2. **Jika overfit gap > 0.050**: params terlalu longgar untuk overnight — pertimbangkan naik ke md=150 atau λ=1.5
3. **Jika return < v7**: params medium tidak memberikan improvement untuk overnight, v7 tetap baseline terbaik
4. **Setelah ada kandidat guard**: jalankan Monte Carlo (block bootstrap 10k paths) sebelum dibandingkan dengan `k2_w25_guard`
