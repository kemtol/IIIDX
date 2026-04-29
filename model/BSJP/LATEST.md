# BSJP — Catatan Pemahaman Sesi Apr 2026

Dokumen ini ditulis sebagai narasi, bukan dokumentasi teknis. Tujuannya supaya siapapun yang baca bisa langsung paham konteks dan keputusan yang sudah dibuat, tanpa harus baca semua kode dan metrics dari awal.

---

## 1. Cara Kerja Pipeline Secara Sederhana

Bayangkan kita punya mesin yang setiap hari belajar dari data pasar, lalu tiap sore mengeluarkan 3 rekomendasi saham yang kemungkinan besar akan naik dari sore ke pagi besok.

Prosesnya ada 4 tahap:

**Tahap 1 — Kumpulkan data mentah (Level 0)**
Setiap hari kita ambil dua jenis data: pertama, aktivitas broker — siapa yang beli dan jual saham apa, dari sumber IPOT. Kedua, data harga OHLCV dari Yahoo Finance. Ini disimpan apa adanya, belum diproses.

**Tahap 2 — Olah jadi fitur (Level 1)**
Data mentah diolah menjadi angka-angka yang bermakna. Misalnya: "broker lokal beli bersih berapa di saham ini 3 hari terakhir?", "berapa kali saham ini gap up dalam 20 hari?". Hasilnya 122 kolom per kombinasi tanggal × broker × saham.

**Tahap 3 — Buat tabel training (Level 2)**
Fitur dari semua broker diagregat per saham per hari. Di sini juga dibuat **label**: apakah saham ini keesokan paginya naik melebihi biaya transaksi (TP) atau justru turun parah (SL)? Label inilah yang akan diajarkan ke model. Hasilnya 234 kolom per kombinasi tanggal × saham.

**Tahap 4 — Training & validasi walk-forward**
Model LightGBM belajar dari data historis, lalu diuji di periode yang belum pernah dilihat sebelumnya (OOT = 100 hari terakhir). Proses ini diulang beberapa kali dengan jendela training yang terus bergeser maju — inilah yang disebut **walk-forward**. Tujuannya agar hasil validasi benar-benar mencerminkan performa di kondisi nyata, bukan di data yang sudah "dihafalkan" model.

Setelah training, model menghasilkan **skor probabilitas** untuk setiap saham setiap hari. Tiga saham dengan skor tertinggi jadi rekomendasi, dengan alokasi modal 60%–30%–10% (rank #1 dapat porsi terbesar karena keyakinan paling tinggi).

---

## 2. Perjalanan Iterasi: Dari v1 Sampai Grid Search

### v1–v7: Belajar sambil jalan

Di awal, setiap versi adalah eksperimen manual — ubah satu hal, lihat hasilnya, ulang. Perlahan-lahan ditemukan hal-hal penting:

- **v5 GAGAL**: Kita filter saham yang terlalu illiquid, tapi justru itu yang jadi sumber edge. Langsung dimatikan.
- **v6**: Fix bug simulasi portofolio. Baseline yang jujur.
- **v7**: Tambah fitur "karakter gap down" — setiap saham punya watak berbeda soal seberapa sering dan parah dia gap down di malam hari. Model yang bisa membaca ini jauh lebih pintar memilih saham yang "aman". Hasilnya: return naik dari +80% ke +103%, drawdown turun dari -56% ke -30%.

**v7 jadi best practice** dan dicatat di edge.md. Angka +102.8% itu konsisten — baik di OOT window lama maupun ketika diukur ulang dengan data terkini (s.d. Apr 2026), hasilnya tetap +102.8%. v7 masih relevan.

### v8–v9: Eksplorasi yang hasilnya lebih rendah dari v7

Setelah v7, ada banyak eksperimen kecil. Semuanya menghasilkan return positif tapi lebih rendah dari v7:
- v8: +66.5% OOT, drawdown -40.6% — lebih volatile dari v7
- v9: +31.8% OOT, drawdown -32.0% — return turun
- v9b: sanity check reproduksi v7 — berhasil (+95.2%), konfirmasi v7 stabil
- v9c: evaluasi year-to-date (hanya 2026) — hasil -42.8%, konfirmasi pasar 2026 lebih keras
- v9d: coba LGBMRanker. Gagal — win rate 44%, return -0.3%

### v10: Dua perubahan besar sekaligus

Di sini ada **pivot metodologi** yang penting. Selama ini model belajar memprediksi return overnight (beli sore, jual pagi). Di v10, dicoba objective baru: **beli di open pagi, jual di close sekitar jam 10:00**. Ini disebut "close10".

Kenapa? Karena di pagi hari setelah market buka, ada lebih banyak fleksibilitas — bisa pasang TP intraday, bisa pilih momen exit yang lebih baik. Exit jam 09:05 itu sangat kaku.

Hasilnya menjanjikan:
- `v10_close10_obj` (k=3, tanpa guard): +226% OOT, tapi drawdown -48% — terlalu dalam.
- `v10_close10_obj_k2_guard`: kurangi ke 2 posisi → drawdown turun ke -34%.
- `v10_close10_obj_k2_w25_guard`: tambah batasan bobot maksimal 25% per saham → drawdown -26%, return +115%, Sharpe 3.57.

**Kandidat ini (`k2_w25_guard`) yang kemudian dijadikan kandidat operasional**, karena satu-satunya yang sudah divalidasi lengkap dengan Monte Carlo 10.000 simulasi.

### v11: Temuan krusial — v7 terlalu dikekang

Sambil jalan, muncul hipotesis baru: **mungkin selama ini model kita terlalu dikekang oleh regularisasi yang terlalu ketat**. v7 pakai `min_data_in_leaf=500` — artinya setiap daun di decision tree harus punya minimal 500 data. Ini sangat konservatif.

Hyperopt dijalankan, dan terbukti: dengan `min_data_in_leaf=50` dan lambda yang lebih longgar, AUC OOT naik dan return melonjak drastis.

Tapi ada trade-off: model yang lebih longgar juga memilih saham yang lebih volatile. Return hariannya naik dari ~1% ke ~4% per hari, tapi volatilitas harian ikut naik dari ~4% ke ~15%. Karena return di-compound setiap hari, selisih vol sekecil itu bisa menghasilkan perbedaan return yang luar biasa besar di akhir 100 hari.

`v11_hyperopt` menunjukkan +1668% di OOT. Angka ini matematis benar, tapi **belum bisa dipercaya** karena belum ada Monte Carlo dan ada risiko hyperparameter overfitting (lihat bagian 4).

### Grid search: bukan metodologi baru, hanya pencarian sistematis

Setelah v11 membuktikan bahwa area params yang lebih longgar itu promising, daripada tebak-tebak satu per satu lagi, dijalankan **grid search** — artinya semua kombinasi params dalam rentang tertentu dicoba sekaligus:

```
min_data_in_leaf  : [50, 80, 100, 200, 500]
lambda_l2         : [0.1, 0.5, 1.0, 2.0]
min_gain_to_split : [0.02, 0.05, 0.1]
```

**Yang penting dipahami**: grid search tidak mengubah fitur, tidak mengubah datamart, tidak mengubah cara walk-forward. Hanya nyari "suhu masak" yang optimal untuk model yang sama.

Hasilnya: `grid_fine/md100_lam1_0_gain0_1` menunjukkan +3257% OOT di atas kertas. Ini kembali ke masalah yang sama — angka besar karena vol harian 14%, bukan karena AUC-nya jauh lebih bagus (cuma naik 0.7%).

---

## 3. Cara Membaca Angka cum_return

Ini sering bikin bingung. Angka `cumulative_net_return` di metrics.json disimpan sebagai **desimal equity compounded**:

- `1.15` → artinya modal tumbuh 115% selama 100 hari OOT
- `16.68` → artinya modal tumbuh 1668%
- `32.57` → artinya modal tumbuh 3257%

Ini sudah diverifikasi silang dengan REPORT.md yang ditulis manual:
- `k2_w25_guard`: metrics.json = `1.152` → REPORT mengatakan "+115.21%" ✓
- `v10_close10_obj`: metrics.json = `2.265` → REPORT mengatakan "+226.47%" ✓

Rumusnya sederhana: setiap hari return portofolio dihitung, lalu di-compound (dikalikan) hari demi hari. Kalau rata-rata return harian 4%, dikali 100 hari secara eksponensial = angka bisa ribuan persen.

---

## 4. Soal Overfitting

### Overfitting model (yang sudah diatasi)

Ini yang umum diketahui: model terlalu hafal data training, tidak bisa generalisasi. Diatasi dengan regularisasi (`lambda`, `min_data_in_leaf`) dan walk-forward validation yang ketat.

### Hyperparameter overfitting (yang perlu diwaspadai dari grid search)

Ini yang lebih halus. Ketika kita menjalankan 16+ kombinasi params dan **memilih yang terbaik berdasarkan OOT metrics**, secara tidak sadar kita sudah memakai OOT sebagai kriteria seleksi. OOT yang seharusnya "steril" jadi ikut masuk ke proses pemilihan model.

Analoginya: bayangkan ujian. Walk-forward = latihan soal. OOT = ujian sesungguhnya. Kalau kita ganti-ganti strategi belajar sampai nilai ujian paling tinggi, lalu bilang "nilai ujian ini mencerminkan kemampuan sesungguhnya" — itu tidak adil, karena ujiannya sudah ikut jadi bagian dari proses belajar.

Cara yang benar adalah: pilih params berdasarkan performa di **periode latihan saja**, lalu evaluasi di OOT **sekali, done**.

Karena itu grid_fine/md100_lam1_0_gain0_1 perlu Monte Carlo dulu sebelum bisa dibandingkan setara dengan kandidat yang sudah tervalidasi.

### AUC berapa yang wajar?

Di financial ML, AUC tinggi justru mencurigakan. Pasar tidak bisa diprediksi dengan akurasi tinggi — kalau bisa, semua orang sudah kaya.

| AUC OOT | Artinya |
|---------|---------|
| < 0.51 | Tidak ada edge, buang |
| 0.51–0.55 | Edge lemah |
| **0.55–0.62** | **Zona sehat — edge nyata tapi realistis** |
| 0.62–0.70 | Mulai curiga, cek lookahead |
| > 0.70 | Hampir pasti ada data leakage |

Semua runs kita ada di 0.577–0.602 — zona yang benar.

Yang lebih penting dari angka AUC:
1. **Overfit gap** (train AUC − OOT AUC) — untuk model dengan 235 fitur seperti punya kita, gap 0.05–0.10 adalah normal (jangan panik kalau lihat angka 0.06–0.08). Gap < 0.03 hanya mungkin untuk model dengan sedikit fitur (< 50).
2. **Stabilitas antar fold** — konsisten lebih baik dari tinggi tapi lompat-lompat
3. **Precision@3** — untuk screener top-3, yang paling relevan adalah seberapa sering rank #1 benar-benar naik, bukan AUC keseluruhan

---

## 5. Perbandingan Frontier Models

Hanya 5 model yang dipertahankan sebagai referensi aktif (sisanya diarsipkan). Berikut perbandingan lengkapnya.

---

### 5A. Ringkasan Cepat

| Model | Strategi | CumRet OOT | MaxDD | WinRate | Vol/hari | Overfit Gap | Monte Carlo |
|-------|----------|-----------|-------|---------|----------|-------------|-------------|
| `bsjp_v7` | overnight | +102.8% | -30.2% | 0.62 | 3.79% | 0.031 ✅ | — |
| `bsjp_v10_close10_obj` | close10 | +226.5% | -48.5% | 0.60 | 5.74% | 0.042 ⚠️ | — |
| `bsjp_v10_close10_obj_k2_w25_guard` | close10 | +115.2% | -26.0% | 0.59 | 3.69% | 0.042 ⚠️ | ✅ done |
| `bsjp_v11_hyperopt` | overnight | +1667.7% | -38.9% | 0.58 | 14.94% | 0.065 ❌ | — |
| `bsjp_grid_fine/md100_lam1_0_gain0_1` | close10 | +3256.5% | -28.2% | 0.67 | 13.95% | 0.073 ❌ | — |

> Overfit gap = Train AUC − OOT AUC. Sehat: < 0.030. Borderline: 0.030–0.050. Concern: > 0.050.

---

### 5B. Detail Per Model

#### `bsjp_v7` — Overnight Baseline

**Karakter**: Model yang konservatif dan stabil. Hasil konsisten antara OOT lama dan terkini (+102.8%). Overfit gap paling sehat (0.031). Drawdown -30.2% masih berat untuk produksi tapi terkontrol.

**LGBM Params**: `min_data_in_leaf=500` | `λ1=2.0, λ2=2.0` | `gain=0.1` | `num_leaves=31` | `depth=4`
*(Sangat conservative — setiap leaf butuh minimal 500 data, regularisasi kuat)*

**Policy**: k=3 | skewed weights 60-30-10 | max_weight=0.34

**Fold AUC Stability**: [0.642, 0.612, 0.597, 0.587] — menurun tapi stabil, trend normal untuk walk-forward.

**Kritik**:
- `min_data_in_leaf=500` terbukti terlalu ketat — tree tidak bisa split di pattern kecil yang valid
- Return +102.8% dengan vol 3.79%/hari → Sharpe sekitar 3.2 (estimasi)
- Best_iteration=35 (sangat rendah) — model berhenti sangat awal, indikasi over-regularized
- Drawdown -30.2% masih cukup dalam untuk modal kecil
- **Belum Monte Carlo**

---

#### `bsjp_v10_close10_obj` — Close10 Baseline (tanpa guard)

**Karakter**: Pivot pertama ke objective close10 (beli open 09:00, jual close 10:00 T+1). Return melonjak ke +226.5% tapi drawdown -48.5% tidak acceptable untuk produksi. Ini adalah "raw" version sebelum policy guard ditambahkan. Dipertahankan sebagai acuan REPORT.md.

**LGBM Params**: `min_data_in_leaf=500` | `λ1=2.0, λ2=2.0` | `gain=0.1` | `num_leaves=31` | `depth=4`
*(Params identik dengan v7 — hanya label/objective yang berubah)*

**Policy**: k=3 | skewed weights 60-30-10 | max_weight=0.34

**Fold AUC Stability**: [0.634, 0.591, 0.574, 0.583] — lebih volatile dari v7, overfit gap 0.042 borderline.

**Kritik**:
- Drawdown -48.5% tidak layak produksi tanpa guard
- Overfit gap 0.042 mulai borderline (threshold 0.030)
- Return tinggi (+226.5%) tapi vol 5.74%/hari lebih besar dari v7
- Model sama persis dengan k2_w25_guard — bedanya hanya policy
- **Belum Monte Carlo**

---

#### `bsjp_v10_close10_obj_k2_w25_guard` — Kandidat Institusional ⭐

**Karakter**: Model identik dengan `v10_close10_obj` di atas (AUC sama, params sama, training sama), hanya policy berbeda: dibatasi maksimal 2 posisi dan maksimal 25% per saham. Efek: drawdown turun drastis dari -48.5% ke -26.0%, return turun dari +226.5% ke +115.2%, tapi **volatilitas harian turun ke 3.69%** — manageable. Satu-satunya yang sudah Monte Carlo.

**LGBM Params**: `min_data_in_leaf=500` | `λ1=2.0, λ2=2.0` | `gain=0.1` | `num_leaves=31` | `depth=4`

**Policy**: k=2 | max_weight=0.25 per posisi | total exposure maksimal 50% modal per hari

**Net expectancy per trade**: 1.67% (tertinggi di antara model conservative — karena hanya ambil 2 picks terbaik)

**Monte Carlo (10.000 paths, block bootstrap)**:
- P(terminal return < 0) dalam 100 hari: **2.70%**
- P(MaxDD ≤ -25%) dalam 252 hari: 43.59%
- Sharpe annualized: **3.57** | Sortino: **5.74**
- VaR 95%: -3.17%/hari | ES 95%: -7.07%/hari

**Kritik**:
- Overfit gap 0.042 — sama dengan v10_close10_obj, borderline
- Return +115.2% lebih rendah dari raw close10 (+226.5%) karena exposure dikurangi
- Exposure 50% per hari artinya 50% modal idle — opportunity cost tinggi
- Model underlying masih over-regularized (md=500) — belum dioptimasi params-nya
- `v11_hyperopt` (overnight, params lebih longgar) menunjukkan bahwa model ini bisa lebih baik

---

#### `bsjp_v11_hyperopt` — Overnight, Params Longgar ⚠️

**Karakter**: Temuan bahwa v7 over-regularized. Dengan params lebih longgar, AUC OOT naik ke 0.595 dan model memilih saham overnight yang lebih volatile — menghasilkan return paper +1667.7%. Tapi vol 14.94%/hari artinya drawdown -38.9% bisa terjadi kapan saja. **Overfit gap 0.065 melampaui threshold 0.050** — ini red flag.

**LGBM Params**: `min_data_in_leaf=50` | `λ1=0.5, λ2=0.5` | `gain=0.01` | `num_leaves=63` | `depth=6`
*(Jauh lebih longgar dari v7 — tree bisa dalam, split lebih sensitif)*

**Policy**: k=3 | skewed weights 60-30-10 | max_weight=0.34

**Fold AUC Stability**: [0.645, 0.628, 0.581, 0.606] — fold 3 turun tajam, menunjukkan ketidakstabilan di periode tertentu.

**Kritik**:
- **Overfit gap 0.065 > 0.050** — model terlalu hafal training data
- Vol 14.94%/hari → drawdown -50%+ realistis terjadi dalam beberapa minggu
- Return +1667.7% datang dari vol tinggi yang ter-compound, bukan dari edge yang jauh lebih besar
- Params hasil hyperopt yang dipilih berdasarkan OOT → ada risiko hyperparameter overfitting
- **Belum Monte Carlo** — angka +1667.7% belum bisa dipercaya

---

#### `bsjp_grid_fine/md100_lam1_0_gain0_1` — Grid Winner (Close10) ⚠️

**Karakter**: Hasil terbaik dari grid search sistematis atas 43 kombinasi params di close10 datamart. AUC 0.584 dengan return paper +3256.5% dan drawdown hanya -28.2% — kelihatannya ideal. Tapi: vol 13.95%/hari, **overfit gap 0.073 tertinggi dari semua frontier models**, dan selection bias karena dipilih berdasarkan OOT performance dari 43 kombinasi.

**LGBM Params**: `min_data_in_leaf=100` | `λ1=1.0, λ2=1.0` | `gain=0.1` | `num_leaves=63` | `depth=6`
*(Sweet spot antara v7 (md=500) dan v11 (md=50) — lebih fleksibel tapi lebih terkontrol dari v11)*

**Policy**: k=3 | skewed weights 60-30-10 | max_weight=0.34

**Fold AUC Stability**: [0.643, 0.589, 0.589, 0.597] — fold 2 dan 3 turun signifikan, tidak stabil.

**Kritik**:
- **Overfit gap 0.073 — tertinggi, melampaui threshold dengan jelas**
- Grid search dijalankan di **close10 datamart**, bukan overnight — params belum pernah ditest di BSJP overnight
- Dipilih berdasarkan OOT performance dari 43 runs → hyperparameter overfitting
- Vol 13.95%/hari setara v11_hyperopt — risiko drawdown dalam sama
- Best_iteration=85 (lebih banyak dari model lain) — model lebih kompleks
- **Belum Monte Carlo** — wajib sebelum dipercaya

---

### 5C. Params Head-to-Head

| Parameter | v7 | v10_close10 | v11_hyperopt | grid_fine |
|-----------|----|-----------|-----------|----|
| `min_data_in_leaf` | 500 | 500 | 50 | **100** |
| `lambda_l1` | 2.0 | 2.0 | 0.5 | **1.0** |
| `lambda_l2` | 2.0 | 2.0 | 0.5 | **1.0** |
| `min_gain_to_split` | 0.1 | 0.1 | 0.01 | **0.1** |
| `num_leaves` | 31 | 31 | 63 | **63** |
| `max_depth` | 4 | 4 | 6 | **6** |
| `best_iteration` | 35 | 41 | 39 | **85** |

> Params v7 dan v10_close10 identik — bedanya hanya label target. Grid_fine adalah "medium" antara konservatif (v7) dan agresif (v11).

---

### 5D. Monte Carlo untuk Top-3 Grid Candidates ✅ (Selesai Apr 2026)

Setelah grid search selesai, 3 kandidat teratas dijalankan Monte Carlo (block bootstrap, block_size=5, 10.000 paths) untuk memvalidasi apakah return paper-nya realistis:

| Kandidat | Alasan Dipilih | MC Median 100d | MC Mean MaxDD | P(MaxDD ≤ -30%) |
|---|---|---|---|---|
| `md100_lam1_5_gain0_02` | Best CumRet (+3442%) | 34.37× | -31.2% | 65.5% |
| `md100_lam1_5_gain0_1` | Best Sharpe (5.109) | 30.57× | **-27.6%** ✅ | **29.1%** ✅ |
| `md100_lam1_0_gain0_1` | Coarse best variant | 33.64× | -32.3% | 51.8% |

**Kesimpulan Monte Carlo:**
- **Pemenang: `md100_lam1_5_gain0_1`** — return median 30.57× (terendah ketiga, tapi drawdown jauh lebih kecil)
- Hanya 29% kemungkinan drawdown > -30% (vs 65% dan 52% untuk dua kandidat lain)
- Mean MaxDD -27.6% — paling rendah di antara ketiga kandidat
- Angka return absolut (30× dalam 100 hari) memang kedengarannya bombastis, tapi itu wajar karena:
  - Return harian rata-rata 4.4% dengan vol 13% — compound effect selama 100 hari
  - Sampel cuma 100 hari — Monte Carlo dengan sampel kecil cenderung ekstrim
  - Yang penting untuk dipercaya adalah **perbandingan antar kandidat**, bukan angka absolutnya

### 5F. ❌ DATA LEAKAGE — Invalidasi Model close10_v10 (Apr 2026)

> **Status: INVALID** — Semua model yang dilatih menggunakan `training_datamart_bsjp_close10_v10.parquet` mengandung data leakage dan harus dianggap tidak valid. Ini termasuk v13 dan seluruh grid search fine/coarse.

#### Kronologi

1. [`v13`](model/BPJS/_ARCH/bpjs_v13/) dilatih menggunakan `training_datamart_bsjp_close10_v10.parquet` — parquet hybrid yang menggabungkan **overnight features (BSJP)** dengan **close10 labels (BPJS)**.
2. Hasil awal: +2698% return, MaxDD -24.6%, Sharpe ~4.56 — terlihat "best-in-class".
3. Setelah investigasi, ditemukan **data leakage struktural**: parquet mengandung kolom target yang seharusnya tidak ada di training datamart.

#### Temuan Forensik

| Kolom | Masalah | Diblokir `OUTCOME_COLS`? |
|-------|---------|--------------------------|
| `exit_price` | 🚨 Harga exit masa depan (target variable) | ✅ Ya — aman |
| `overnight_return` | 🚨 Return overnight (label BSJP sebagai feature BPJS) | ✅ Ya — aman |
| `close_ret_last1h` | ⚠️ Return 1 jam terakhir hari T — context overnight, **tidak relevan untuk entry 09:00 T+1** | **❌ TIDAK** — lolos sebagai feature #1 (gain 37,361) |

**Akar masalah:**
- [`build_close10_objective_datamart.py`](edges/bsjp_overnight_sl2/scripts/build_close10_objective_datamart.py:1) mengambil base overnight datamart dan **hanya mengganti label**, TIDAK membersihkan kolom-kolom yang bersifat overnight-timing.
- Kolom seperti `close_ret_last1h`, `ret_1h_to_close`, dan seluruh keluarga fitur closing momentum tetap ada — padahal untuk entry 09:00 T+1, harga close T sudah diketahui dan tidak informatif untuk prediksi intraday 09:00→10:00.

#### Dampak ke Semua Model

| Model | Datamart | Status |
|-------|----------|--------|
| `bpjs_v13` 🆕 | close10_v10 | **❌ INVALID** — data leakage |
| Grid fine `md100_lam1_0_gain0_1` | close10_v10 | **❌ TERKONTAMINASI** — overfit gap 0.073 mungkin sebagian dari leakage |
| Grid fine `md100_lam1_5_gain0_1` | close10_v10 | **❌ TERKONTAMINASI** — MC winner, perlu di-re-run |
| Grid fine `md100_lam1_5_gain0_02` | close10_v10 | **❌ TERKONTAMINASI** |
| `k2_w25_guard` | close10 (v1) | **⚠️ INVESTIGASI** — perlu dicek apakah parquet v1 juga bermasalah |
| `v10_close10_obj` | close10 (v1) | **⚠️ INVESTIGASI** — sama dengan di atas |

#### Tindakan

- [x] v13 → [`_ARCH/bpjs_v13/`](model/BPJS/_ARCH/bpjs_v13/) ✅ Archived
- [x] Dokumentasi temuan di LATEST.md (bagian ini)
- [ ] **Prioritas: Train BSJP asli (overnight)** dengan datamart clean [`training_datamart_bsjp_overnight.parquet`](data/Level_2_Datamart/training_datamart_bsjp_overnight.parquet) — 730 hari, 248 kolom, IHSG MA ✅
- [ ] Perbaiki pipeline: `build_close10_objective_datamart.py` harus membersihkan kolom target dan overnight-context features
- [ ] Grid search perlu di-re-run dengan datamart yang terbukti clean

---

## 6. Status Saat Ini & Prioritas Selanjutnya

### ❌ `bpjs_v13` — INVALID (Data Leakage)

> Model ini telah diarchive ke [`_ARCH/bpjs_v13/`](model/BPJS/_ARCH/bpjs_v13/). Tidak valid karena data leakage pada datamart close10_v10. Lihat [Bagian 5F](#5f--data-leakage--invalidasi-model-close10_v10-apr-2026) untuk forensik lengkap.

### ✅ Kandidat mature: `v10_close10_obj_k2_w25_guard` (baseline)

Sudah divalidasi lengkap — tetap jadi acuan karena vol lebih rendah (3.69%/hari):
- Return OOT 100 hari: **+115.21%**
- Max drawdown: **-26.04%** | Sharpe: **3.57**
- Win rate: **59%** | Volatilitas harian: **3.69%**
- Monte Carlo: peluang modal negatif di 100 hari hanya **2.70%**

> **Catatan:** Model ini juga dilatih di close10 datamart (v1, bukan v10). Perlu diverifikasi apakah parquet v1 juga mengandung kolom leakage. Jika ya, model ini juga perlu diinvestigasi ulang.

### ⚠️ Monte Carlo Grid — Terkontaminasi Data Leakage

Monte Carlo untuk top-3 grid candidates sudah dijalankan, **namun seluruh grid search dilatih menggunakan `close10_v10` parquet yang terbukti mengandung data leakage**.

Hasil MC tetap tercatat sebagai referensi, **tetapi tidak bisa dijadikan dasar keputusan**:

| Kandidat | MC Median 100d | Status |
|----------|---------------|--------|
| `md100_lam1_5_gain0_02` | 34.37× | ❌ Terkontaminasi |
| `md100_lam1_5_gain0_1` | 30.57× | ❌ Terkontaminasi — MC winner |
| `md100_lam1_0_gain0_1` | 33.64× | ❌ Terkontaminasi |

Artifact Monte Carlo tetap disimpan:
- [`md100_lam1_5_gain0_02/monte_carlo/`](model/BSJP/bsjp_grid_fine/md100_lam1_5_gain0_02/monte_carlo/)
- [`md100_lam1_5_gain0_1/monte_carlo/`](model/BSJP/bsjp_grid_fine/md100_lam1_5_gain0_1/monte_carlo/)
- [`md100_lam1_0_gain0_1/monte_carlo/`](model/BSJP/bsjp_grid_fine/md100_lam1_0_gain0_1/monte_carlo/)

### Agenda Selanjutnya (prioritas)

1. **🟢 Train BSJP asli (overnight)** — `min_data_in_leaf=100`, `lambda_l1/l2=1.5`, `min_gain_to_split=0.1`, `k2_w25_guard`. Pakai datamart clean [`training_datamart_bsjp_overnight.parquet`](data/Level_2_Datamart/training_datamart_bsjp_overnight.parquet) — 730 hari, 248 kolom, termasuk IHSG MA. Ini adalah model BSJP pertama yang memanfaatkan temuan params grid (md100_lam1_5) tanpa data leakage.
2. **📋 Update edge.md** — dokumentasi data leakage, archive v13, dan hasil BSJP asli setelah training.
3. **🔍 Investigasi `k2_w25_guard`** — verifikasi apakah close10 v1 parquet juga bermasalah. Jika clean, model ini tetap valid sebagai baseline.
4. **📈 Paper trade** `k2_w25_guard` — tetap jalan selama investigasi berlangsung, karena ini yang paling mature.
5. **🔄 Re-run grid search** — jika pipeline data sudah diperbaiki, grid search perlu diulang dengan datamart yang clean.

---

## 7. Sesion Apr 27: Execution Model, Bug Fix, Modular Feature Loading

### 7A. `--execution-model` Parameter

Ditambahkan CLI argument baru di [`train_lightgbm.py`](edges/bsjp_overnight_sl2/scripts/train_lightgbm.py:197):

```bash
--execution-model {limit,market}  # default: limit
```

**`limit` (default):** Fill di `entry_price`/`exit_price` dengan fixed broker cost 0.4% round trip. Perilaku original.

**`market`:** Stress test bid-ask spread berdasarkan aturan tick IDX. Buy di ask (premium), sell di bid (discount). Round trip spread = 2x `estimate_spread_frac()`.

#### IDX Tick Rules (di `estimate_spread_frac()`)

| Rentang Harga | Tick | Multiplier | Spread (ticks) |
|---------------|------|------------|----------------|
| < Rp200       | 1    | 2.5        | 2-3 ticks      |
| Rp200-500     | 2    | 2.0        | 2-4 ticks      |
| Rp500-2k      | 5    | 1.5        | 1-2 ticks      |
| Rp2k-5k       | 10   | 1.5        | 1-2 ticks      |
| > Rp5k        | 25   | 1.0        | 1 tick         |

One-way spread fraction = `(tick * multiplier) / price`. Round trip cost = 2x nilai ini.

#### Market Order Simulation

Di [`simulate_portfolio()`](edges/bsjp_overnight_sl2/scripts/train_lightgbm.py:610):

```
Buy on ask:  entry_price * (1 + spread_frac)    # beli lebih mahal
Sell on bid: exit_price * (1 - spread_frac)      # jual lebih murah
```

Spread cost additive dengan existing broker cost (0.2% entry + 0.2% exit).

### 7B. 🔴 Critical Bug Fix: `--feature-prune-top-n` Default

**Bug:** Default `--feature-prune-top-n 35` di [`line 143`](edges/bsjp_overnight_sl2/scripts/train_lightgbm.py:143) secara diam-diam membatasi model ke **35 features hardcoded**, sementara v17 memakai 249 features.

**Dampak:**

| Model | Features | AUC | Result |
|-------|----------|-----|--------|
| v18_market_order (20d OOT) | 35 | 0.488 | FAIL |
| v18_control_limit (20d OOT) | 35 | 0.537 | PASS tapi misleading |
| v18_market_100d (100d OOT) | 35 | 0.504 | FAIL |
| v18_market_100d_fix (100d OOT) | **249** | **0.642** | **PASS (+0.94%/day)** |

**Fix:** Default diubah dari `35` → `0`. Help text: "0 = disable pruning, use all numeric columns".

**Lessons:**
- Selalu verifikasi feature count di `metrics.json` setelah training
- Default harus 0 (pakai semua numeric columns) kecuali sengaja pruning
- File fallback `ranked_candidates` path bisa degrade kualitas model secara diam-diam

### 7C. v18 Market Order Stress Test Results

Dibandingkan dengan v17 (limit order, 249 features, 100d OOT):

| Metric | v17 (limit) | v18_market_100d_fix (market) | Delta |
|--------|-------------|------------------------------|-------|
| AUC OOT | 0.642 | 0.642 | 0.0 |
| CumRet OOT | +3.44%/day | +0.94%/day | -73% of alpha |
| MaxDD | -20.8% | -17.1% | better risk mgmt |
| Sharpe | 3.54 | 2.02 | still healthy |
| P(loss 100d) from MC | — | 5.3% | acceptable |

Spread mengkonsumsi 73% alpha harian, tapi model masih PASS — edge masih cukup kuat untuk nutup spread cost.

### 7D. Modular Feature Loading — Phase 1 & 2 ✅ (Selesai Apr 2026)

**Akar masalah:** [`generate_datamart.py`](edges/bsjp_overnight_sl2/scripts/generate_datamart.py) adalah monolithic ETL — untuk menambah feature baru harus edit script, merge, rebuild L2 full (~3 menit), risk of breaking changes.

**Solusi:** Pisahkan setiap feature group ke parquet modular.

#### Arsitektur Baru (Sudah Berjalan)

```
data/Level_1_Features/modules/               # 7 modules, 245 MB total
├── closing_momentum_features.parquet        # 879 KB, 3 feats
├── overnight_history_features.parquet       # 4.5 MB, 30 feats
├── broker_aggregate_features.parquet        # 213 MB, 188 feats
├── cvd_features.parquet                    # 7.6 MB, 6 feats
├── global_indices_features.parquet         # 117 KB, 15 feats (date-only grain)
├── stockbit_xl_features.parquet            # 778 KB, 4 feats
└── vwap_features.parquet                   # 19 MB, 6 feats
```

#### Cara Kerja

[`generate_datamart.py`](edges/bsjp_overnight_sl2/scripts/generate_datamart.py) menulis 7 module parquets di akhir setiap function `build_*()` jika `--modules-dir` diberikan (opsional, `modules_enabled` guard untuk backward compat).

[`train_lightgbm.py`](edges/bsjp_overnight_sl2/scripts/train_lightgbm.py) dengan flag `--feature-modules-dir` hanya membaca 8 core columns dari monolithic L2, lalu LEFT JOIN otomatis semua `*_features.parquet`:

```python
# Actual code (load_features_from_modules, line 866-890)
for fpath in sorted(modules_dir.glob("*_features.parquet")):
    module = pd.read_parquet(fpath)
    has_ticker = "ticker" in module.columns
    on = ["date", "ticker"] if has_ticker else ["date"]
    df = df.merge(module, on=on, how="left")
```

Grain auto-detection: jika module punya kolom `ticker` → merge `(date, ticker)`, jika tidak (contoh: global indices) → merge `(date,)` saja.

#### Zero Quality Tradeoff ✅ Terverifikasi

Kedua jalur (monolithic vs modular) pakai:
- Source data yang sama (L0/L1)
- Feature engineering logic yang sama
- LEFT JOIN yang sama pada (date, ticker)
- `float32` dtype yang sama
- `shift(1)` no-lookahead yang sama

**Hasil verifikasi:** `pd.testing.assert_frame_equal()` pada 108,698 rows — **252/252 features identical**, termasuk `vix_prev_close` dari global_indices (date-only grain).
- Feature set: 252 of 252 match ✓
- Row count: 108,698 both paths ✓
- Column count: 260 both paths ✓

#### Performance Target — Actual Results

| Metric | Monolithic (Before) | Modular (After) | Improvement |
|--------|--------------------|-----------------|-------------|
| Load L2 for training | ~3 min (read full 700MB) | **~5-10s** (read 8 cols + join modules) | **~20x** |
| Add new feature | edit + rebuild | drop parquet only | instant |
| Incremental update | ~3 min | ~5-10s | ~20x |

#### Implementation Phases — Status

1. **Phase 1 — Extract ✅** [`generate_datamart.py`](edges/bsjp_overnight_sl2/scripts/generate_datamart.py) — 7 module writes, `modules_enabled` guard, backward compatible. All modules produced successfully (245 MB total).

2. **Phase 2 — Load ✅** [`train_lightgbm.py`](edges/bsjp_overnight_sl2/scripts/train_lightgbm.py) — `--feature-modules-dir` arg, `load_features_from_modules()` function, grain auto-detection, `CORE_MODEL_COLS` (8 columns). Syntax verified + feature set verified (252/252 match).

3. **Phase 3 — Refactor:** Optionally make generate_datamart.py read modules instead of inline building. (Belum dimulai)

4. **Phase 4 — Sanity check:** Full rebuild + random check + document. (Partial — feature set verified, byte-level comparison pending fresh monolithic)

#### Model Iterations Summary (Updated)

| Model | Execution | OOT Days | Features | AUC | CumRet/Day | Status |
|-------|-----------|----------|----------|-----|-----------|---|
| v17_compound | limit | 100 | 249 | 0.642 | +3.44% | ✅ Reference |
| v18_market_order | market | 20 | 35 | 0.488 | — | ❌ FAIL (bug) |
| v18_control_limit | limit | 20 | 35 | 0.537 | +0.50% | ⚠️ Bug (35 feat) |
| v18_market_100d | market | 100 | 35 | 0.504 | — | ❌ FAIL (bug) |
| **v18_market_100d_fix** | **market** | **100** | **249** | **0.642** | **+0.94%** | **✅ PASS** |
