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

### 7E. v18 Close10 Rebuild — OLD Objective Recovered (Apr 29)

Forensics showed that the OLD `training_datamart_bsjp_overnight.parquet` used by v18 was not true overnight. Its `label_name` is `bsjp_close10_sl2`, with entry `close@15 T` and exit `open@10 T+1`. The NEW overnight datamart uses `open@09 T+1`, causing large label drift.

Rebuilt close10 with current `generate_datamart.py --exit-hour 10`, then filtered to the OLD `(date,ticker)` universe:

- `data/Level_2_Datamart/training_datamart_bsjp_close10_rebuild.parquet`: full close10 rebuild, 424,909 rows, too broad for v18 reproduction.
- `data/Level_2_Datamart/training_datamart_bsjp_close10_rebuild_v18like.parquet`: OLD universe, 108,698 rows, `bsjp_close10_sl2`.
- Core drift vs OLD on v18like: `exit_price` 0 changed rows, `label_sl2` 0 changed rows, `entry_price`/`label_tp` 1 changed row only.

Trained `model/BSJP/bsjp_v18_close10_rebuild/` with OLD v18 params:

| Model | Rows | Best Iter | OOT AUC | Mean Daily Net | CumNet | MaxDD | Status |
|-------|-----:|----------:|--------:|---------------:|-------:|------:|---|
| OLD `bsjp_v18_market_100d_fix` backup | 108,698 | 305 | 0.6417 | +0.94% | 1.238x | -30.6% | Reference |
| `bsjp_v18_close10_rebuild` | 108,698 | 408 | 0.6501 | +1.30% | 2.126x | -35.6% | PASS, MC done |

Monte Carlo (10,000 paths, block=5):

| Horizon | Median Terminal | P(loss) | Mean MaxDD | P(MaxDD ≤ -30%) |
|---|---:|---:|---:|---:|
| 100d | 3.38x | 2.0% | -25.2% | 24.8% |
| 252d | 22.98x | 0.05% | -32.2% | 54.6% |

Interpretation: the close10 objective is recovered. Rebuild is stronger on AUC/return, but OOT and Monte Carlo drawdown risk is material; paper-trade policy should be explicit before promotion.

### 7F. v18 Close10 Policy Guard — k3/w25 (Apr 30)

Policy sensitivity on `valid_predictions.parquet` showed raw k3/w34 has strong return but concentrated tail risk. A lower concentration cap keeps the same model and top-3 structure while reducing drawdown.

Recommended paper-trade artifact:

- `model/BSJP/bsjp_v18_close10_rebuild_policy_w25/`
- Same model predictions as `bsjp_v18_close10_rebuild`
- Policy: threshold mode, `p_cut=0.035`, adaptive threshold q=0.85, `max_positions=3`, `max_weight_per_name=0.25`, market execution

| Policy | Mean Daily Net | CumNet | MaxDD | Worst Day | Days <= -5% | Days <= -8% |
|---|---:|---:|---:|---:|---:|---:|
| Raw k3/w34 | +1.30% | 2.126x | -35.6% | -10.3% | 5 | 1 |
| Guard k3/w25 | +0.98% | 1.431x | -28.2% | -7.6% | 2 | 0 |

Monte Carlo for guard k3/w25:

| Horizon | Median Terminal | P(loss) | Mean MaxDD | P(MaxDD <= -30%) |
|---|---:|---:|---:|---:|
| 100d | 2.57x | 1.78% | -19.6% | 7.61% |
| 252d | 11.34x | 0.04% | -25.2% | 22.0% |

### 7G. current_v18_fixed Forensics — Hybrid Model Not Promotable (Apr 30)

`model/BSJP/bsjp_v18_market_100d_fix/` looked very strong on the latest windows, but forensic checks showed its datamart is a hybrid:

- 105,102 rows labeled `bsjp_close10_sl2`
- 5,134 rows labeled `bsjp_overnight_sl2`
- The OOT tail includes favorable `NEW_extra` rows, so the model/policy is not a clean close10 or clean overnight experiment.

To test whether the strength came from row universe rather than mixed labels, a clean fixed-universe close10 datamart was built:

- `data/Level_2_Datamart/training_datamart_bsjp_close10_fixed_universe.parquet`
- 105,793 rows from the fixed `(date,ticker)` universe that have valid close10 labels
- 4,443 fixed rows were dropped because they only existed in the overnight side and had no clean close10 match
- All remaining rows use `label_name=bsjp_close10_sl2`

Training result:

| Model | Rows | Best Iter | OOT AUC | 100d CumNet | 50d CumNet | 20d CumNet | MaxDD 100d | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `bsjp_v18_market_100d_fix` | 110,236 | 28 | 0.6431 | +217.5% | +113.5% | +65.2% | -11.3% | Hybrid, not promotable |
| `bsjp_v18_close10_fixed_universe` | 105,793 | 21 | 0.6376 | -41.6% | -38.8% | -15.3% | -50.2% | Clean test failed |
| `bsjp_v18_close10_rebuild_policy_w25` | 108,698 | 408 | 0.6501 | +143.1% | -4.4% | +1.7% | -28.2% | Clean paper-trade candidate |

Conclusion: `current_v18_fixed` should be treated as a diagnostic clue only. Once the mixed-label contamination is removed, the fixed-universe idea fails badly. The defensible candidate remains `bsjp_v18_close10_rebuild_policy_w25`, with the caveat that its last 50d/20d windows are not impressive.

### 7H. v19 Preclose14 Cost-Aware Baseline (Apr 30)

This run fixes two issues in v18 close10:

1. Operational lookahead: same-day close15/EOD features are blacklisted.
2. Objective mismatch: labels are trained from market-cost-aware outcome while `overnight_return` stays gross for one-time market execution simulation.

Artifacts:

- `data/Level_1_Features/modules/preclose14_features.parquet`: 470,187 rows, 21 features, `(date,ticker)` grain.
- `data/Level_2_Datamart/training_datamart_bsjp_close10_costaware.parquet`: thin core datamart, 108,698 rows.
- `model/BSJP/bsjp_v19_close10_preclose14_costaware/`

Blacklist preset `preclose14` removes:

`close_ret_last1h`, `close_vs_open_day`, `close_range_pct`, `close_to_vwap`, `open_pm_to_vwap_am`, `vwap_trend`, `last_hour_above_vwap`, `close_drive`, `vol_above_vwap_pct`.

Replacement feature family includes pre-14:59 price momentum, VWAP, volume/liquidity, and execution-cost proxy features such as `pre14_spread_cost_est` and `pre14_market_cost_est`.

Cost-aware labels are much stricter:

| Metric | Old Gross Label | Cost-Aware Label |
|---|---:|---:|
| TP rate | 34.6% | 8.6% |
| SL2 rate | — | 68.7% |
| Mean market cost estimate | — | 3.67% |

Training result:

| Model | Features | Best Iter | OOT AUC | OOT AUCPR | 100d CumNet | 50d CumNet | 20d CumNet | MaxDD 100d | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `bsjp_v18_close10_rebuild_policy_w25` | 249 | 408 | 0.6501 | 0.4949 | +143.1% | -4.4% | +1.7% | -28.2% | Operational lookahead |
| `bsjp_v19_close10_preclose14_costaware` | 261 | 7 | 0.7433 | 0.2711 | +19.1% | -22.0% | +11.9% | -48.9% | Executable baseline, not promotable |

Interpretation: removing close15/EOD features and making labels cost-aware reduces small-cap spread exposure (`price<500` top3 falls from 53.0% to 23.7% on 100d), but the model is now dominated by cost proxies and has too little realized edge. The baseline is executable but not yet suitable for promotion due to severe drawdown and weak 50d behavior.

### 7I. v19b Gross-Label Preclose14 + Execution Filters (Apr 30)

Follow-up test: keep the old gross close10 label, but keep the preclose14 feature blacklist. This isolates whether alpha survives after removing close15/EOD timing lookahead. Execution remains market HAKA/HAKI.

Trainer patch added policy filters:

- `--max-pre14-market-cost-est`
- `--min-entry-price-filter`
- `--min-pre14-turnover-until14`

Runs:

| Model | Label | Policy Filter | OOT AUC | 100d CumNet | 50d CumNet | 20d CumNet | MaxDD 100d | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| `bsjp_v19b_close10_preclose14_grosslabel` | gross `>0.4%` | none | 0.5205 | -50.7% | -61.2% | -14.7% | -72.4% | FAIL |
| `bsjp_v19b_close10_preclose14_grosslabel_costfilter35` | gross `>0.4%` | `pre14_cost<=3.5%` | 0.5205 | +4.3% | -37.6% | +4.1% | -61.1% | PASS but weak |
| `bsjp_v19b_close10_preclose14_grosslabel_costfilter30_p500` | gross `>0.4%` | `pre14_cost<=3.0%`, `entry>=500` | 0.5205 | +39.9% | -16.9% | -4.5% | -34.4% | Best v19b, not promotable |

Interpretation: cost/price filters materially improve preclose14 gross-label results, but the model's OOT AUC is nearly random and recent 50d/20d remain weak. This supports the hypothesis that v18's strong edge depended heavily on close15/EOD operational lookahead. The best v19b variant is useful as a research baseline, not as a trading candidate.

### 7J. v19c Preclose14 Volume/Turnover + Small Grid (Apr 30)

The `preclose14_features.parquet` module was expanded from 21 to 51 features to test executable proxies for participation before 14:59:

- Relative volume vs historical daily volume: `pre14_am_volume_to_dailyvol_{5,10,20}d`, `pre14_until14_volume_to_dailyvol_{5,10,20}d`
- Relative turnover vs historical daily turnover: `pre14_am_turnover_to_dailyturnover_{5,10,20}d`, `pre14_until14_turnover_to_dailyturnover_{5,10,20}d`
- Session 2 vs session 1 activity: `pre14_pm_am_volume_ratio`, `pre14_pm_am_volume_rate_ratio`, `pre14_pm_am_turnover_ratio`, `pre14_pm_am_turnover_rate_ratio`
- PM price-action additions: `pre14_pm_high_vs_close11`, `pre14_pm_high_close_spread`, `pre14_pm_close_near_high`, open-low distance flags

Univariate read: the new volume/turnover features have weak standalone signal, but some are used by the model. In `md100_l2=1.5`, relative volume/turnover contributes about 2.3% of total gain. Strongest relative-volume feature is `pre14_am_volume_to_dailyvol_10d`.

Baseline `bsjp_v19c_close10_preclose14_volturn_grosslabel_costfilter30_p500`:

| Model | Features | OOT AUC | 100d CumNet | 50d CumNet | 20d CumNet | MaxDD 100d |
|---|---:|---:|---:|---:|---:|---:|
| `v19b costfilter30_p500` | 261 | 0.5205 | +39.9% | -16.9% | -4.5% | -34.4% |
| `v19c vol/turn costfilter30_p500` | 290 | 0.5314 | +36.7% | -13.6% | -0.2% | -28.8% |

Interpretation: volume/turnover additions improve risk and recent-window behavior, but reduce 100d return slightly. They help defensiveness more than raw alpha.

Small 9-run hyperparameter sanity grid for v19c:

- `min_data_in_leaf`: 50, 100, 200
- `lambda_l2`: 0.5, 1.5, 3.0
- `min_gain_to_split`: fixed 0.05
- Same execution filters: `pre14_market_cost_est <= 0.030`, `entry_price >= 500`
- Summary: `model/BSJP/grid_v19c_small/summary.csv`

Sorted by walk-forward CV mean daily net:

| Run | CV/day | OOT 100d | MaxDD | 50d | 20d | OOT AUC |
|---|---:|---:|---:|---:|---:|---:|
| `md50_l2=1.5` | +1.24% | +82.8% | -30.7% | -8.0% | +0.5% | 0.5340 |
| `md50_l2=3.0` | +1.24% | +85.8% | -36.2% | -12.3% | -2.5% | 0.5332 |
| `md50_l2=0.5` | +1.12% | +36.7% | -28.8% | -13.6% | -0.2% | 0.5314 |
| `md100_l2=1.5` | +1.00% | +97.5% | -29.9% | +0.0% | +7.5% | 0.5308 |

Anti-overfit read: do not pick solely by OOT 100d. The best CV run is `md50_l2=1.5`, while the more balanced research candidate is `md100_l2=1.5` because 50d/20d are healthier without chasing the top OOT number. Both remain research candidates only: best iteration is still 7 and OOT AUC is around 0.53, so the executable preclose alpha is thin.

Current research candidate for observation:

- `model/BSJP/grid_v19c_small/md100_l21.5_gain0_05/`
- 100d +97.5%, 50d +0.04%, 20d +7.48%, MaxDD -29.93%, AUC 0.5308

Do not promote v19c yet. Next useful idea is finding an executable proxy for the unavailable close15/EOD push, not broad hyperparameter search.

### 7K. v19d ORB Preclose14 Proxy (Apr 30)

Added ORB-style features to `preclose14_features.parquet`, expanding the module to 68 features. Strict 09:00 ORB coverage was too sparse in Yahoo 1h history, so ORB uses the first available morning bar from hours 9/10/11 and records `pre14_orb_source_hour` for audit.

ORB feature examples:

- `pre14_orb_range_pct`
- `pre14_close_vs_orb_mid`
- `pre14_close_vs_orb_high`
- `pre14_close_vs_orb_low`
- `pre14_orb_position`
- `pre14_orb_breakout_strength`
- `pre14_pm_high_vs_orb_high`
- `pre14_pm_high_breakout_strength`
- `pre14_orb_body_pct`, wick features, and ORB breakout flags

Univariate sanity:

- `pre14_orb_range_pct`: AUC_abs 0.5348
- `pre14_close_vs_orb_low`: AUC_abs 0.5279
- Several ORB position/breakout features have weak standalone signal, but became useful in the model.

Trained `model/BSJP/bsjp_v19d_close10_preclose14_orb_md100_l21.5/` using the prior balanced params:

- `min_data_in_leaf=100`
- `lambda_l2=1.5`
- `min_gain_to_split=0.05`
- `pre14_market_cost_est <= 0.030`
- `entry_price >= 500`
- market execution, max weight 25%

Result:

| Model | OOT AUC | Best Iter | 100d CumNet | 50d CumNet | 20d CumNet | MaxDD 100d |
|---|---:|---:|---:|---:|---:|---:|
| v19c grid `md100_l2=1.5` | 0.5308 | 7 | +97.5% | +0.0% | +7.5% | -29.9% |
| **v19d ORB `md100_l2=1.5`** | **0.5412** | **5** | **+148.6%** | **+24.3%** | **+9.9%** | **-29.0%** |
| v18 close10 w25 | 0.6501 | 408 | +143.1% | -4.4% | +1.7% | -28.2% |

Top v19d features now include ORB structure:

- `pre14_orb_position`
- `pre14_close_vs_orb_mid`
- `pre14_close_vs_orb_low`
- `pre14_orb_breakout_strength`
- `pre14_orb_range_pct`

Interpretation: ORB is the first executable proxy family that materially improves both OOT and recent windows. This is still a research candidate, not production-ready: best iteration remains very low (5), and selection is not yet validated by a fresh walk-forward/grid after adding ORB. But v19d is now the best executable preclose14 candidate to observe.

### 7L. v19d ARA-like Execution Filter (Apr 30)

Added preclose14 ARA proxy columns to `preclose14_features.parquet`, expanding the module to 73 features:

- `pre14_prev_close`
- `pre14_return_from_prev_close`
- `pre14_ara_limit_pct`
- `pre14_ara_distance_pct`
- `pre14_is_ara_like`

The ARA proxy uses the standard upper auto-rejection bands by reference price: 35% for Rp50-Rp200, 25% for >Rp200-Rp5,000, and 20% for >Rp5,000, with a 0.5% buffer. These columns are `POLICY_ONLY_COLS` in `train_lightgbm.py`, so they are carried for simulation but blocked from LightGBM feature selection.

Trainer patch:

- `--exclude-pre14-ara-like`
- policy-only exclusion for ARA proxy columns
- policy backtest carries ARA proxy columns in `valid_predictions.parquet`

Run: `model/BSJP/bsjp_v19d_close10_preclose14_orb_md100_l21.5_noara/`

| Model | ARA Filter | OOT AUC | Best Iter | 100d CumNet | MaxDD 100d | Status |
|---|---:|---:|---:|---:|---:|---|
| `v19d ORB base` | no | 0.5412 | 5 | +148.6% | -29.0% | PASS |
| `v19d ORB noARA` | yes | 0.5412 | 5 | -42.3% | -44.2% | FAIL |

Impact audit:

- Policy-like OOT picks before ARA filter: 264 rows across 96 days
- ARA-like picked rows: 83 rows across 62 days
- Removed ARA-like picks had mean gross return +8.04% and mean net return +6.27%
- Kept non-ARA picks had mean gross return +0.89% and mean net return -0.85%
- Summary artifact: `_LOG/pre14_ara_filter_impact_20260430.csv`
- Full training log: `_LOG/train_bsjp_v19d_orb_noara_20260430.log`

Interpretation: the v19d ORB portfolio edge is concentrated in names that are already near ARA by 14:59. That is not model leakage, but it is an execution realism problem: the backtest assumes market buy execution at the entry price even when the order book may have no obtainable offer. A hard no-ARA filter makes the candidate set unattractive, so the next research step should not be a simple exclude. More realistic choices are fill-probability haircut, queue/liquidity-aware sizing, or a separate "ARA continuation but likely fillable" filter using order book/offer availability if such data exists.

### 7M. Active Research Baseline Reframe (May 1)

The active BSJP research baselines are now split into two branches:

| Branch | Artifact | Role | 100d CumNet | 50d CumNet | 20d CumNet | MaxDD 100d |
|---|---|---|---:|---:|---:|---:|
| ARA continuation | `bsjp_v19d_close10_preclose14_orb_md100_l21.5` | Main research baseline; alpha exists but fillability is unresolved | +148.6% | +24.3% | +9.9% | -29.0% |
| Non-ARA continuation | `bsjp_v19d_close10_preclose14_orb_md100_l21.5_noara` | Negative-control baseline after hard excluding near-ARA names | -42.3% | n/a | n/a | -44.2% |

`bsjp_v18_close10_rebuild_policy_w25` is no longer an active research baseline. It stays as a historical recovered-v18 reference only, because it depends on close15/EOD-style features that are not operationally available before a 14:59 decision.

Research implication:

- ARA continuation branch: keep the signal and solve fillability/queue realism.
- Non-ARA branch: current hard-filter baseline is weak; needs new alpha/features rather than simply reusing v19d after excluding ARA-like names.

### 7N. ARA State / Release-Wick Diagnostic (May 1)

Refined ARA execution realism beyond a hard near-ARA exclusion. Added hourly ARA-state features to `preclose14_features.parquet`, expanding the module to 90 features:

- `pre14_ara_price`
- `pre14_ara_touched`
- `pre14_ara_touch_hour`
- `pre14_ara_touched_bar_count`
- `pre14_ara_release_wick_count`
- `pre14_ara_release_wick_ratio`
- `pre14_ara_release_wick_depth_max`
- `pre14_ara_release_wick_depth_mean`
- `pre14_ara_close_at_ara_count`
- `pre14_ara_flat_ohlc_count`
- `pre14_ara_last_bar_locked`
- `pre14_ara_locked_proxy`
- `pre14_ara_touched_released`

Hourly proxy logic:

- ARA touched: hourly high reaches ARA price within one tick.
- Release wick: the same hourly bar touches ARA but low trades below ARA by more than one tick.
- Locked proxy: ARA touched, no release wick through pre-14 bars, and last bar closes at ARA.

Diagnostic artifacts:

- `_LOG/pre14_ara_state_bucket_diagnostic_20260501.csv`
- `_LOG/pre14_ara_state_selected_examples_20260501.csv`

Selected top-policy bucket read, using existing v19d predictions joined with new ARA-state features:

| Bucket | Rows | Days | Mean Overnight | Median Overnight | TP Hit | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| `ara_touched_single_release` | 43 | 39 | +11.99% | +13.01% | 79.1% | Best current fillable-ARA bucket |
| `near_ara_not_touched_0_3pct` | 5 | 5 | +9.37% | +6.80% | 80.0% | Near-ARA but not locked; sample small |
| `momentum_near_ara_3_8pct` | 17 | 16 | +3.47% | +1.31% | 52.9% | Tradable momentum bucket |
| `ara_touched_repeated_release` | 56 | 45 | +1.41% | -1.22% | 41.1% | Repeated wick may indicate inventory release/distribution |
| `non_ara_far_gt8pct` | 142 | 81 | +0.94% | +0.00% | 45.8% | Weak baseline continuation |
| `ara_touched_locked_proxy` | 1 | 1 | +21.79% | +21.79% | 100.0% | Too sparse to judge |

Interpretation: hard no-ARA was too crude. The better first split is not ARA vs non-ARA, but ARA touched with single release wick vs repeated release wick vs near-ARA not touched vs far non-ARA. Initial evidence supports the user's execution hypothesis: a wick below ARA means there was some tradable release, but repeated release wicks may be distribution rather than clean continuation.

### 7O. ARA-State Policy Simulation Without Retrain (May 1)

Ran exact portfolio simulations using existing v19d predictions joined with the new ARA-state features. This is a policy-only test; model scores are unchanged.

Artifacts:

- `_LOG/pre14_ara_state_policy_sim_summary_20260501.csv`
- `_LOG/pre14_ara_state_policy_sim_daily_20260501.csv`
- `_LOG/pre14_ara_state_policy_sim_trades_20260501.csv`
- `_LOG/pre14_ara_state_policy_veto_summary_20260501.csv`
- `_LOG/pre14_ara_state_policy_veto_daily_20260501.csv`

Two policy mechanics were tested:

1. **Pre-filter mode**: apply bucket eligibility before ranking/threshold. This matches `train_lightgbm.py` filter semantics.
2. **Veto mode**: score/select with the original v19d policy, then discard unwanted buckets without re-ranking or reweighting. This is closer to a practical execution veto layered on top of the current model.

Pre-filter sanity check:

| Variant | Trading Days | Positions | 100D | 50D | 20D | MaxDD |
|---|---:|---:|---:|---:|---:|---:|
| `all_v19d_base` | 96 | 264 | +148.6% | +24.3% | +9.9% | -29.0% |
| `single_release_near_momentum_0_8pct` | 35 | 35 | +135.0% | +51.7% | +14.8% | -4.7% |
| `single_release_plus_near_0_3pct` | 27 | 27 | +86.0% | +35.7% | +8.4% | -3.4% |
| `single_release_only` | 26 | 26 | +54.8% | +25.1% | +2.4% | -3.6% |
| `repeated_release_only` | 19 | 19 | +15.1% | -2.3% | -3.3% | -8.7% |
| `far_non_ara_only` | 94 | 231 | -56.4% | -31.2% | -9.5% | -58.3% |

Veto-mode read:

| Variant | Trading Days | Positions | 100D | 50D | 20D | MaxDD |
|---|---:|---:|---:|---:|---:|---:|
| `veto_keep_single_near_momentum_0_8` | 51 | 65 | +248.6% | +59.0% | +18.7% | -3.2% |
| `veto_keep_single_plus_near_0_3` | 42 | 48 | +222.7% | +66.1% | +23.0% | -3.2% |
| `veto_keep_single_release_only` | 39 | 43 | +193.7% | +61.1% | +16.2% | -3.6% |
| `veto_exclude_repeated_locked` | 94 | 207 | +161.4% | +40.6% | +19.6% | -19.0% |
| `veto_repeated_only` | 45 | 56 | -9.2% | -15.8% | -8.3% | -20.6% |
| `veto_far_nonara_only` | 81 | 142 | -24.8% | -11.3% | +0.9% | -32.0% |

Interpretation: the best policy read is not "exclude ARA", but **keep ARA touched with a single release wick, keep near-ARA not touched, optionally keep momentum 3-8% from ARA, and veto repeated release / far non-ARA**. Repeated release is now clearly a weak bucket, supporting the inventory-distribution hypothesis.

Caveat: this is OOT diagnostic policy selection, so do not promote directly. The next validation should freeze this bucket rule and test it on a fresh split or walk-forward policy selection.

### 7P. Frozen ARA-State Walk-Forward Validation (May 1)

Added `edges/bsjp_overnight_sl2/scripts/validate_ara_state_policy.py` to validate frozen ARA-state veto rules outside the final 100d OOT diagnostic. The script retrains v19d-style models over the original 4 pre-OOT walk-forward folds and applies the frozen bucket rules to each validation fold. ARA-state columns are explicitly policy-only and blocked from model features.

Artifacts:

- `_LOG/validate_ara_state_policy_20260501.log`
- `_LOG/pre14_ara_state_policy_walkforward_summary_20260501.csv`
- `_LOG/pre14_ara_state_policy_walkforward_daily_20260501.csv`
- `_LOG/pre14_ara_state_policy_walkforward_trades_20260501.csv`
- `_LOG/pre14_ara_state_policy_walkforward_predictions_20260501.parquet`
- `_LOG/pre14_ara_state_policy_walkforward_by_fold_20260501.csv`

Walk-forward summary over 80 validation days:

| Variant | Trading Days | Positions | CumNet | MaxDD | Mean Daily | Net/Trade |
|---|---:|---:|---:|---:|---:|---:|
| `wf_all_base` | 75 | 219 | +174.1% | -13.8% | +1.36% | +1.99% |
| `wf_veto_single_near_momentum_0_8` | 41 | 51 | +111.4% | -7.4% | +0.98% | +6.12% |
| `wf_veto_single_plus_near_0_3` | 37 | 45 | +92.6% | -7.4% | +0.86% | +6.09% |
| `wf_veto_exclude_repeated_locked` | 73 | 188 | +76.1% | -14.8% | +0.77% | +1.32% |
| `wf_veto_single_release_only` | 33 | 39 | +64.4% | -7.4% | +0.65% | +5.33% |
| `wf_veto_repeated_only` | 26 | 31 | +57.2% | -6.8% | +0.59% | +6.06% |
| `wf_veto_far_nonara_only` | 69 | 137 | -16.2% | -22.3% | -0.20% | -0.47% |

Fold stability:

- `wf_veto_single_near_momentum_0_8` is positive in all 4 folds: +28.9%, +22.4%, +14.3%, +17.3%.
- `wf_veto_single_plus_near_0_3` is positive in all 4 folds: +31.1%, +10.9%, +13.0%, +17.3%.
- `wf_veto_far_nonara_only` is weak/negative overall: -16.2% total and -22.3% MaxDD.

Interpretation: frozen ARA-state rules validate as a **risk and trade-quality filter**. They reduce drawdown and lift net expectancy per trade materially, but in pre-OOT walk-forward they do not beat the full base policy on cumulative return. The cleaner branch is `single_plus_near_0_3`; the broader branch `single_near_momentum_0_8` trades more and has higher cumulative return. Both are valid research candidates for the next ARA-continuation variant, but still not production candidates.

### 7Q. v20 Clean ARA-Continuation Policy Artifact (May 1)

Packaged the recommended clean policy as a model-like artifact:

- `model/BSJP/bsjp_v20_ara_continuation_state_policy_clean/`
- Builder: `edges/bsjp_overnight_sl2/scripts/package_ara_state_policy.py`
- Source model: `model/BSJP/bsjp_v19d_close10_preclose14_orb_md100_l21.5/`
- Policy layer only: no retraining; same v19d model file copied into the artifact.

Frozen policy:

- Keep `ara_touched_single_release`
- Keep `near_ara_not_touched_0_3pct`
- Veto repeated release, far non-ARA, locked proxy, and other buckets
- Mechanic: veto after original v19d selection, no re-ranking and no reweighting

Artifact files:

- `metrics.json`
- `model_lightgbm_opening_tp3.txt`
- `feature_importance.csv`
- `portfolio_daily.parquet`
- `policy_trades.parquet`
- `valid_predictions_with_ara_state.parquet`
- `policy_summary.csv`
- `pnl_100d.png`, `pnl_50d.png`, `pnl_20d.png`

OOT policy metrics:

| Artifact | Trading Days | Positions | 100D | 50D | 20D | MaxDD | Net/Trade |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v19d full base` | 96 | 264 | +148.6% | +24.3% | +9.9% | -29.0% | +1.52% |
| `v20 clean policy` | 42 | 48 | +222.6% | +66.1% | +23.0% | -3.2% | +10.05% |

`policy_summary.csv` also includes comparison rows:

| Policy | 100D | 50D | 20D | MaxDD | Positions |
|---|---:|---:|---:|---:|---:|
| `aggressive` | +248.6% | +59.0% | +18.7% | -3.2% | 65 |
| `clean` | +222.6% | +66.1% | +23.0% | -3.2% | 48 |
| `single_release_only` | +193.7% | +61.1% | +16.2% | -3.6% | 43 |
| `all` | +148.6% | +24.3% | +9.9% | -29.0% | 264 |

Status: **research baseline artifact, not production**. The policy has strong OOT diagnostics and positive walk-forward validation as a risk filter, but the rule was derived during research and still needs paper-trade/fillability observation before production use.

### 7R. No-Touch ARA Alternative Direction (May 1)

Follow-up execution concern: even `ara_touched_single_release` may still be hard to execute live because it already touched ARA. Tested no-touch subsets from existing v19d selections as a diagnostic:

Artifact:

- `_LOG/pre14_ara_state_no_touch_alternatives_20260501.csv`

No-touch bucket results:

| Bucket | OOT 100D | WF Validation | OOT MaxDD | OOT Trades | Read |
|---|---:|---:|---:|---:|---|
| `near_ara_not_touched_0_3pct` | +9.6% | +17.4% | -2.7% | 5 | Most executable, very sparse |
| `momentum_3_8_not_touched` | +8.2% | +9.7% | -4.4% | 17 | More trades, weaker edge |
| `near_momentum_0_8_not_touched` | +18.6% | +28.7% | -7.0% | 22 | Candidate no-touch branch, needs more filtering |
| `all_not_touched` | -11.0% | +7.5% | -30.4% | 164 | Too noisy |

Interpretation: the cleanest operational bucket is **near ARA but not touched**, because it should still be tradable without ARA queue risk. The problem is sparse signal count. Next research should start from a **v20 no-touch clean** branch:

- Primary seed: `near_ara_not_touched_0_3pct`
- Broader candidate: `near_ara_not_touched_0_3pct + momentum_near_ara_3_8pct`
- Goal: add filters/features to increase trade count without admitting ARA-touched names.

Do not replace `bsjp_v20_ara_continuation_state_policy_clean` yet. Treat no-touch as the next branch to develop.

---

## 8. Go Inference Build (May 2026)

Multi-session effort to bring the Go inference binary to feature parity with v19d + v20 ARA policy.

### 8A. Architecture

Go binary (`inferences/bsjp/golang/cmd/bsjp/`) has three commands:
- `bootstrap`: parquet L0 → DuckDB features_store (one-time, ~315 columns)
- `fetch`: incremental fetch for latest date (reads L0 parquet, computes features)
- `predict`: load model, score tickers, apply ARA policy, output top-K

Feature modules mirror Python structure, all writing to `features_store` table in DuckDB.

### 8B. Features Implemented

| Module | Go File | Cols | Calibration |
|--------|---------|------|-------------|
| Global | `global.go` | 15 | PERFECT (15/15) |
| Momentum | `momentum.go` | 4 | PERFECT (4/4) |
| yf_daily | `yfdaily.go` | 58 | PERFECT (58/58) after `.JK` fix |
| Overnight | `overnight.go` | ~30 | gapdown_freq epsilon |
| CVD | `cvd.go` (new) | 6 | via DuckDB SQL window |
| Preclose14 | `preclose14.go` (new) | 115 | 85/115 PERFECT; ~30 volume cols differ |
| Broker | `broker_agg.go` | ~187 | not calibrated, CGO UPDATE bug |

**Total: 360 cols/ticker for v19d**, of which 306 are model features, 5 trees.

### 8C. Bugs Found and Fixed

1. **yf_daily `.JK` suffix** — `parquet-go` reads tickers as `BBRI.JK` but momentums already strips suffix. Added `strings.TrimSuffix` to `yfdaily.go`. Was ALL NaN, now PERFECT 58/58.
2. **Duplicate key on UpsertDate** — Switched from `DELETE+INSERT` to DuckDB-compatible `INSERT OR REPLACE`.
3. **Flow order bug** — broker/CVD UPDATEs ran BEFORE `UpsertDate`, so `INSERT OR REPLACE` overwrote with NULL. Fixed: upsert first, then broker UPDATE, then CVD UPDATE.
4. **Dedup logic** — Duplicate tickers in features_store caused `Duplicate key` errors. Added merge-before-upsert in cmdFetch.

### 8D. Preclose14 Volume Calibration Gap

115 preclose14 columns: 85 match exactly, ~30 volume/turnover columns have small numeric diffs. Root cause: `parquet-go` `parquet.ReadFile` reads volume column differently from Python `pd.read_parquet`. Same bar filtering logic (NaN check on OHLC, preclose hours). Price-based features (close14, VWAP, orb_position) are exact match.

### 8E. v20 ARA-State Policy in Go Predict

Implemented in `main.go` `cmdPredict()`. Auto-activates when variant name contains `v19d` or `v20`:

- **Keep**: `ara_touched_single_release` (ARA touched, exactly 1 release wick)
- **Keep**: `near_ara_not_touched_0_3pct` (within 0-3% of ARA, not touched)
- **Veto**: repeated release, far non-ARA, locked proxy
- **Mechanic**: veto after scoring, no re-ranking, fallback to unfiltered if all vetoed

First run: 756 picks → 2 picks (KOBX, NIRO). Sparse, matching v20 research "48 positions in 42 days" finding.

### 8F. Remaining Gaps

1. Preclose14 volume calibration (~30 cols) — investigate parquet-go null handling
2. Overnight epsilon — gapdown_freq likely `<=` vs `<` comparison
3. Broker calibration (~187 cols) — complex DuckDB SQL, need systematic comparison
4. Stockbit/XL, VWAP — not implemented (~10 features)
5. Live yfinance intraday fetcher — current `bsjp fetch` reads existing parquet files only

---

## 9. v23 Clean ARA-History Research Baseline (2026-05-09)

After the broker same-day leakage audit, v19d/v20 headline returns are no longer trusted as proof of edge. P0 patched residual CVD lookahead, Step A added `ara_history_features.parquet`, and P0.5 rebuilt `preclose14_features.parquet` to full coverage.

Step C trained three comparable clean runs on the v19d-like close10 core:

| Model | Feature intent | Features | Best iter | WF AUC | OOT AUC | OOT cum net | MaxDD | Mean/day | Read |
|---|---|---:|---:|---|---:|---:|---:|---:|---|
| `v23a_ara_history_pre14state_fullcoverage_clean` | ARA-history + pre14 ARA-state as model features | 336 | 9 | 0.575/0.568/0.521/0.562 | 0.5433 | +107.9% | -30.4% | +0.84% | Better AUC, worse drawdown |
| `v23b_ara_history_only_fullcoverage_clean` | ARA-history only; ARA-state remains policy/backtest-only | 314 | 5 | 0.603/0.560/0.545/0.538 | 0.5283 | +194.1% | -17.9% | +1.17% | Best first portfolio profile |
| `v23c_no_ara_history_state_fullcoverage_clean` | No ARA-history, no ARA-state | 302 | 8 | 0.576/0.569/0.548/0.548 | 0.5273 | +115.7% | -26.2% | +0.85% | Clean full-coverage baseline still positive |

Common setup: `training_datamart_bsjp_close10_rebuild_v18like.parquet`, `--feature-modules-dir data/Level_1_Features/modules`, preclose14 blacklist, market execution, `pre14_market_cost_est <= 0.030`, `entry_price >= 500`, k=3, max weight 25%, p_cut=0.035, OOT 100 days, `min_data_in_leaf=100`, `lambda_l1=0.5`, `lambda_l2=1.5`, `min_gain_to_split=0.05`.

Interpretation:

- v23 is a research baseline, not production.
- `v23b` is the current best candidate to investigate next, but OOT AUC is weak and best iteration is only 5. Treat the +194% OOT as promising but fragile.
- `v23a` shows that opening pre14 ARA-state to the model improves global OOT AUC, but does not improve the practical trading profile in this first run.
- `v23c` staying positive means the improvement is not exclusively from ARA-history; full-coverage preclose14/regime features are also doing real work.
- Next step should be robustness and risk diagnostics on v23b: worst-day review, bucket attribution, policy variants, no-broker ablation, then Monte Carlo only if those checks pass.

### 9A. v23b Residual `yf_daily` Leak Audit

Follow-up audit found that old `v23b_ara_history_only_fullcoverage_clean` still inherited same-day `yf_daily_*` values through `broker_aggregate_features.parquet`. This was smaller than the original broker leak but still invalid for a pre14 decision point because final daily high/low/close are not known before close.

Confirmed before patch:

- `yf_daily_open_mean`: 207,949 / 208,062 comparable rows matched same-day daily open
- `yf_daily_high_mean`: 207,949 / 208,062 matched same-day daily high
- `yf_daily_low_mean`: 207,949 / 208,062 matched same-day daily low
- `yf_daily_close_mean`: 207,949 / 208,062 matched same-day daily close
- `yf_daily_range_pct_mean`: 208,062 / 208,062 matched same-day daily range

Patch:

- `edges/bpjs_opening_tp3/scripts/generate_datamart.py::build_feature_aggregate()` now shifts current-row L1 numeric families per `(broker, ticker)` before date/ticker aggregation:
  - `flow_`, `ctx_`, `tfl_`, `yf_daily_`, `yf_1h_`, `yf_4h_`, `pc_`, `has_yf_`
- z/velocity are recomputed from shifted T-1 flow vs shifted historical baseline.
- Regenerated `data/Level_1_Features/modules/broker_aggregate_features.parquet`.
- Backup: `data/Level_1_Features/modules/broker_aggregate_features.parquet.pre_yfdaily_t1_fix.bak`

Exact reproduction audit after patch:

- `yf_daily_high_mean`: 204,220 / 204,220 match shifted L1 aggregate
- `yf_daily_close_mean`: 204,220 / 204,220 match shifted L1 aggregate
- `yf_daily_range_pct_mean`: 204,220 / 204,220 match shifted L1 aggregate
- `flow_total_net_buy_mean`: 245,698 / 245,698 match shifted L1 aggregate
- `ctx_broker_market_share_mean`: 245,698 / 245,698 match shifted L1 aggregate
- `tfl_net_buy_z_20_mean`: 176,360 / 176,360 match shifted L1 aggregate

Replacement clean candidate:

| Model | OOT AUC | OOT cum net | MaxDD | Mean/day | Best iter | Overfit gap | Read |
|---|---:|---:|---:|---:|---:|---:|---|
| `v23b_t1audit_clean` | 0.5346 | +207.9% | -24.2% | +1.22% | 5 | 0.0616 | Clean against residual `yf_daily` leak; still overfit-risk candidate |

Feature-family contribution in `v23b_t1audit_clean`:

- pre14 intraday: 67.1% gain
- macro prev-close: 26.7%
- ARA-history T-1: 2.6%
- `yf_daily` now T-1-or-older: 1.8%
- broker aggregate T-1: 1.6%
- CVD T-1: 0.07%

Interpretation: old v23b should not be cited anymore. Use `v23b_t1audit_clean` as the current candidate. It survives the residual leak patch, but best iteration 5 and AUC overfit gap 0.0616 mean it remains research-only until robustness diagnostics pass.

### 9B. Train-Time Pruning Check

Feature pruning was tested as train-time exclusion only. No module files or columns were deleted.

| Model | Features | Best iter | WF AUC | OOT AUC | OOT cum net | MaxDD | Mean/day | Overfit gap |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `v23b_t1audit_clean` | 315 | 5 | 0.586/0.554/0.539/0.546 | 0.5346 | +207.9% | -24.2% | +1.22% | 0.0616 |
| `v23d_pruned78_t1audit_clean` | 78 | 13 | 0.574/0.571/0.554/0.560 | 0.5349 | +173.6% | -29.6% | +1.11% | 0.0702 |
| `v23e_pruned50_t1audit_clean` | 50 | 7 | 0.569/0.572/0.546/0.550 | 0.5352 | +91.8% | -22.4% | +0.74% | 0.0666 |

Artifacts:

- `_LOG/v23b_t1audit_clean_nonzero78_features.txt`
- `_LOG/v23_pruning_comparison_20260509.csv`

Read: naive top-N pruning did not improve the model. AUC stayed similar, but portfolio quality worsened. Keep `v23b_t1audit_clean` as the current research candidate; next improvement should come from robustness diagnostics, bucket/policy controls, and fold-stable feature selection rather than simple one-run top-N pruning.

### 9C. Full No-Lookahead Audit Gate

Added a reproducible audit script:

- `edges/bsjp_overnight_sl2/scripts/audit_v23_no_lookahead.py`

This script is read-only and checks feature-set leakage, pre14 cutoff reconstruction, macro timing, broker-shift exactness, and ARA-history exact rebuild.

During this audit, one small stale-data mismatch was found in `ara_history_features.parquet`:

- `max_return_5d_tminus1`: 1 mismatch out of 1,848,026 comparable rows
- Row: `BJBR`, 2026-04-22
- module value: 0.011834
- fresh raw rebuild value: 0.005952

Cause: ARA-history module was stale vs current `yfinance_daily`. Regenerated:

- `data/Level_1_Features/modules/ara_history_features.parquet`
- New shape: 1,850,329 rows x 14 cols
- Date range: 2011-06-03 -> 2026-05-08

After regeneration, ARA-history exact rebuild audit passed:

- `was_ara_tminus1`: 1,849,554 / 1,849,554
- `ara_count_5d`: 1,849,554 / 1,849,554
- `ara_count_20d`: 1,849,554 / 1,849,554
- `days_since_last_ara`: 952,639 / 952,639
- `last_ara_return`: 952,639 / 952,639
- `max_return_5d_tminus1`: 1,848,779 / 1,848,779
- `max_return_20d_tminus1`: 1,848,779 / 1,848,779

Retrained the current clean candidate on the fully current modules:

| Model | OOT AUC | OOT cum net | MaxDD | Mean/day | Best iter | Overfit gap |
|---|---:|---:|---:|---:|---:|---:|
| `v23b_t1audit2_clean` | 0.5346 | +207.9% | -24.2% | +1.22% | 5 | 0.0616 |

Final audit artifact:

- `_LOG/v23b_t1audit2_clean_no_lookahead_audit_20260509.json`

Hard audit failures are all false:

- `feature_blacklist_present`: false
- `outcome_like_present`: false
- `policy_only_unblocked`: false
- `pre14_reconstruction_failed`: false
- `pre14_selected_after_14`: false
- `broker_shift_failed`: false

Selected proof points:

- pre14 selected max hour = 14
- pre14 selected bars after 14 = 0
- 822,887 rows after 14 existed in raw/session data but were not selected
- `pre14_prev_close`: 471,400 / 471,400 match expected previous close
- broker shifted exact matches:
  - `yf_daily_high_mean`: 204,220 / 204,220
  - `yf_daily_close_mean`: 204,220 / 204,220
  - `yf_daily_range_pct_mean`: 204,220 / 204,220
  - `flow_total_net_buy_mean`: 245,698 / 245,698
  - `ctx_broker_market_share_mean`: 245,698 / 245,698
  - `tfl_net_buy_z_20_mean`: 176,360 / 176,360

Read: current candidate is now `v23b_t1audit2_clean`. No hard leakage/lookahead failure remains in the current audit scope. Remaining objections are robustness/overfit, not confirmed leakage.

### 9D. Robustness Diagnostics: Worst Days + Policy Micro-Grid

Added robustness scripts:

- `edges/bsjp_overnight_sl2/scripts/_v23_policy_tools.py`
- `edges/bsjp_overnight_sl2/scripts/analyze_worst_days_v23.py`
- `edges/bsjp_overnight_sl2/scripts/stress_policy_grid_v23.py`

Both diagnostics enforce a baseline reconstruction gate: simulated daily portfolio must match `model/BSJP/v23b_t1audit2_clean/portfolio_daily.parquet` exactly before any attribution or policy result is trusted.

Worst-day attribution:

- Artifacts:
  - `_LOG/v23b_t1audit2_clean_worst_days_20260509.csv`
  - `_LOG/v23b_t1audit2_clean_worst_day_pick_attribution_20260509.csv`
  - `_LOG/v23b_t1audit2_clean_worst_day_bucket_summary_20260509.csv`
  - `_LOG/v23b_t1audit2_clean_worst_day_price_summary_20260509.csv`
  - `_LOG/v23b_t1audit2_clean_worst_day_summary_20260509.json`
- Baseline match: max daily net-return diff 0.0; position mismatch days 0.
- Worst/loss days selected: 24 days (`net_return <= -2%` union bottom 10).

Worst-day weighted loss by bucket:

| Bucket | Trades | Weighted net |
|---|---:|---:|
| `far_or_other` | 43 | -0.4284 |
| `touched_repeated_release` | 15 | -0.2958 |
| `touched_single_release` | 7 | -0.0996 |
| `virgin_near_not_touched` | 1 | -0.0347 |
| `momentum_near_3_8_not_touched` | 4 | -0.0341 |

Worst-day weighted loss by price tier:

| Price tier | Trades | Weighted net |
|---|---:|---:|
| `500_2000` | 46 | -0.5758 |
| `2000_5000` | 18 | -0.2688 |
| `gt5000` | 6 | -0.0480 |

Policy micro-grid:

- Full grid was too slow with the initial pandas simulator, so first-pass `--micro-grid` was used.
- Artifact: `_LOG/v23b_t1audit2_clean_policy_stress_micro_grid_20260509.csv`
- Baseline match: max daily net-return diff 0.0; position mismatch days 0.

| Policy | Active days | OOT cum net | MaxDD | Worst 30D | Read |
|---|---:|---:|---:|---:|---|
| Baseline: k=3, max weight 25%, cost cap 3%, q=.85 | 96 | +207.9% | -24.2% | -10.1% | Current policy |
| k=2, max weight 25%, cost cap 3%, q=.85 | 96 | +255.0% | -17.9% | +7.6% | Best first robust candidate |
| k=2, max weight 20%, cost cap 3%, q=.90, tick p95 veto | 96 | +184.7% | -13.5% | +8.2% | Lower-DD candidate |
| k=2, max weight 20%, cost cap 3%, q=.85 | 96 | +179.1% | -14.5% | +6.3% | Simple lower-risk candidate |

Important negative result:

- `exclude_pre14_ara_like=True` performed badly:
  - baseline-like k=3 version: -42.4% cum, MaxDD -56.0%
  - k=2/cost 2.5% version: -17.6% cum, MaxDD -36.6%
- Therefore a blunt ARA-like hard veto should not be used as-is; it removes too much useful signal.

Current robustness read: the first policy improvement candidate is **k=2 / max weight 25% / cost cap 3% / adaptive q=.85**. It improves OOT return and drawdown while keeping active days unchanged, but it still needs rolling OOT validation before replacing the baseline policy.

### 9E. k=2 Quick-Win Validation

Added:

- `edges/bsjp_overnight_sl2/scripts/evaluate_policy_windows_v23.py`
- policy args to `edges/bsjp_overnight_sl2/scripts/analyze_worst_days_v23.py`

Rolling OOT subwindow artifacts:

- `_LOG/v23b_t1audit2_clean_policy_daily_compare_20260509.csv`
- `_LOG/v23b_t1audit2_clean_policy_summary_compare_20260509.csv`
- `_LOG/v23b_t1audit2_clean_policy_rolling_windows_20260509.csv`
- `_LOG/v23b_t1audit2_clean_policy_rolling_window_summary_20260509.csv`
- `_LOG/v23b_t1audit2_clean_policy_rolling_report_20260509.json`

Policy comparison:

| Policy | Active days | Cum net | MaxDD | Vol daily | Net/trade |
|---|---:|---:|---:|---:|---:|
| baseline k=3/w25/q85 | 96 | +207.9% | -24.2% | 4.25% | 1.76% |
| quickwin k=2/w25/q85 | 96 | +255.0% | -17.9% | 3.94% | 2.89% |
| lowerDD k=2/w20/q90 | 96 | +184.7% | -13.5% | 3.14% | 3.01% |

Rolling subwindow read:

| Metric | baseline k=3 | k=2/w25 | k=2/w20/q90 |
|---|---:|---:|---:|
| 7D positive-rate | 73.4% | 84.0% | 85.1% |
| Worst 20D | -15.8% | -3.5% | -1.5% |
| Worst 30D | -10.1% | +5.9% | +6.1% |
| Worst 60D | +30.6% | +56.3% | +46.5% |

k=2/w25 worst-day attribution:

- Artifacts:
  - `_LOG/v23b_t1audit2_clean_quickwin_k2_w25_worst_days_20260509.csv`
  - `_LOG/v23b_t1audit2_clean_quickwin_k2_w25_worst_day_pick_attribution_20260509.csv`
  - `_LOG/v23b_t1audit2_clean_quickwin_k2_w25_worst_day_bucket_summary_20260509.csv`
  - `_LOG/v23b_t1audit2_clean_quickwin_k2_w25_worst_day_price_summary_20260509.csv`
  - `_LOG/v23b_t1audit2_clean_quickwin_k2_w25_worst_day_summary_20260509.json`
- Worst/loss days: 19 vs baseline 24.
- Worst total weighted net: -0.6786 vs baseline -0.8925.
- Largest k=2 worst buckets:
  - `touched_repeated_release`: 14 trades, weighted net -0.3038
  - `far_or_other`: 14 trades, weighted net -0.2207
  - `touched_single_release`: 6 trades, weighted net -0.1204
- Largest k=2 worst price tier:
  - `500_2000`: 26 trades, weighted net -0.5196

Read: **k=2/w25/q85 is now the current quick-win policy candidate** for `v23b_t1audit2_clean`. It improves return, MaxDD, volatility, net/trade, worst 20D/30D/60D, and keeps active days unchanged. Conservative alternative is k=2/w20/q90. This is still final-OOT/subwindow validation, not yet a full rolling-retrain validation.

### 9F. Calendar Anchor Correction: Latest Closed PnL Is Not 2026-05-09

User corrected the reporting anchor: "last N trading days" should be anchored from the current calendar date, 2026-05-09, not from the locked v23 OOT artifact end date.

Data coverage check:

| Artifact | Max date | Read |
|---|---:|---|
| `model/BSJP/v23b_t1audit2_clean/valid_predictions.parquet` | 2026-04-23 | Locked 100D OOT artifact |
| `data/Level_0_Raw/yfinance_daily.parquet` | 2026-05-08 | Raw daily is current through Friday |
| `data/Level_0_Raw/yfinance_1h.parquet` | 2026-05-08 ~09:03 WIB | No 10:xx candle for 2026-05-08 yet |
| `data/Level_1_Features/modules/preclose14_features.parquet` | 2026-05-07 | Preclose features available through 2026-05-07 |
| Broker aggregate/CVD modules | 2026-04-23 | Broker-family module coverage gap after locked OOT |

Therefore, the latest locally computable **closed close10 trade** is:

- Entry date: 2026-05-06
- Exit date: 2026-05-07 10:xx

It is not valid to report closed PnL through 2026-05-09:

- 2026-05-09 is Saturday.
- Entry 2026-05-08 exits on Monday 2026-05-11, so it is not closed.
- The local 1h file also lacks the 2026-05-08 10:xx candle needed to close a 2026-05-07 entry.

Artifacts created for the provisional calendar extension:

- `_LOG/v23_liveextend_close10_20260424_20260508.parquet`
- `_LOG/v23b_t1audit2_clean_combined_predictions_to_20260506_dedup.parquet`
- `_LOG/v23b_t1audit2_clean_k2w25_daily_to_20260506_dedup.csv`
- `_LOG/v23b_t1audit2_clean_k2w25_trades_to_20260506_dedup.csv`
- `_LOG/v23_restore_full_modules_close10_20260509.parquet`

Important hygiene note: the incremental datamart generator wrote some module parquets in partial/incremental shape. A full-history restore run was executed afterward to avoid leaving `data/Level_1_Features/modules/` in that partial state.

Latest local module coverage after restore:

| Module | Date range |
|---|---|
| `ara_history_features.parquet` | 2011-06-03 -> 2026-05-08 |
| `broker_aggregate_features.parquet` | 2024-10-01 -> 2026-04-23 |
| `closing_momentum_features.parquet` | 2023-04-03 -> 2026-05-08 |
| `cvd_features.parquet` | 2024-10-01 -> 2026-04-23 |
| `global_indices_features.parquet` | 2021-05-06 -> 2026-05-07 |
| `overnight_history_features.parquet` | 2024-11-05 -> 2026-05-07 |
| `preclose14_features.parquet` | 2023-03-06 -> 2026-05-07 |
| `session_intensity_features.parquet` | 2023-03-05 -> 2026-05-08 |
| `stockbit_xl_features.parquet` | 2026-05-06 -> 2026-05-07 |
| `vwap_features.parquet` | 2023-03-06 -> 2026-04-24 |

Corrected Rp10m summaries using quick-win k=2/w25/q85, anchored to latest locally computable closed date 2026-05-06:

| Window | Period | Ending capital | PnL | Return |
|---|---|---:|---:|---:|
| 7 trading days | 2026-04-27 -> 2026-05-06 | Rp11,131,121 | +Rp1,131,121 | +11.31% |
| 30 trading days | 2026-03-17 -> 2026-05-06 | Rp12,247,629 | +Rp2,247,629 | +22.48% |
| 90 trading days | 2025-12-11 -> 2026-05-06 | Rp27,150,569 | +Rp17,150,569 | +171.51% |

Last 7 closed trading days:

| Date | Picks | Daily net |
|---|---|---:|
| 2026-04-27 | IFSH, SMMT | +4.08% |
| 2026-04-28 | KONI, ALKA | +3.18% |
| 2026-04-29 | APIC, HBAT | +4.35% |
| 2026-04-30 | GGRM, SONA | -0.87% |
| 2026-05-04 | HERO, UDNG | +1.30% |
| 2026-05-05 | UDNG, KONI | +1.38% |
| 2026-05-06 | ABDA, GGRM | -2.42% |

Read: this extension is useful for user-facing calendar intuition, but it is **not equivalent to locked OOT**. Treat all post-2026-04-23 performance as provisional until broker aggregate/CVD module coverage is brought forward and the extension is regenerated cleanly.
