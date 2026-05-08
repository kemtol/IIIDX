# v21 — Clean Broker Retrain Progress

## Goal

Hilangkan lookahead pada keluarga broker features (`sq_/xc_/yp_/pd_*`, `flow_*_sum`, `flow_*_mean`, `tfl_*`), retrain v19d-style model (close10 + preclose14 ORB), dan compare honest OOT vs angka existing yang inflated.

## Why this work was triggered (user direction)

Sesi 2026-05-07/08:
1. User observe banyak saham ARA jadi rekomendasi, secara operasional sulit di-fill.
2. Saya gambar arsitektur — terlihat v19d edge memang concentrated di ARA-like names (`program.md:320` confirm), v20 policy hanya membuang 2 dari 4 ARA buckets.
3. User cek: solid? overfit? — saya jelaskan ada 3 caveat: (a) cross-sectional features `sq/xc/yp/pd` belum di-audit lookahead, (b) v20 policy dipilih lihat OOT (potensi OOT-overfit), (c) operasional sulit (fillability gap).
4. Diskusi PnL vs MaxDD: v19d+v20 backtest +222.6% / DD −3.2%. Drop dari v19d-only DD −29% → −3.2% mencurigakan (typical pattern OOT-tuning).
5. User minta rekomendasi → saya urutkan #1 audit lookahead (foundational), #2 fill simulation, #3 paper-trading 40+ hari, #4 fillability gate, #5 frozen walk-forward.
6. User confirm jalankan #1 audit → ditemukan **smoking gun**: training pakai broker same-day data tapi inference Go pakai T-1. Comment `generate_datamart.py:618` klaim "T-1 safe" tapi kode tidak.
7. User konfirmasi: fix + train v21 baru, **step by step perlahan**, commit dulu sebelum mulai, tulis progress per step di markdown ini.

## Background

| Layer | Behavior |
|---|---|
| L0 broksum | `net_val` per `date=D` = aktivitas hari D, published EOD ~17:00 (post-decision) |
| L1 build (`generate_broksum_datamart.py:649`) | `.shift(1)` cuma untuk kolom `_lag1` & rolling mean/std baselines. Raw `flow_total_net_buy` un-shifted |
| L1 z/velocity (`generate_broksum_datamart.py:661-662`) | numerator pakai today's `df[col]`, denominator pakai shifted mean/std → **leaky** |
| Training merge (`generate_datamart.py:639`) | merge `(date, ticker)` tanpa shift; comment line 618 klaim "T-1 safe" tapi kode tidak |
| Inference Go (`main.go:284`) | `WHERE date < targetDate` → broker date strict T−1 → **safe** |

Akibatnya, model belajar dari same-day broker flow di training tapi cuma terima T−1 di production. Backtest +148.6% / +222.6% di program.md kemungkinan inflated.

## Plan (step-by-step)

- [x] **Step 1.** Patch broker shift di `build_feature_aggregate` (`edges/bpjs_opening_tp3/scripts/generate_datamart.py:476`). 13 raw broker columns di-shift `(.shift(1))` per `(broker, ticker)` sebelum aggregate; `tfl_net_buy_z_*` / `tfl_net_buy_velocity_*` direcompute dari `tfl_net_buy_lag1` vs rolling baseline. **L1 broksum_datamart tidak di-touch** — fix dilakukan di layer training-build, inference Go sudah pakai T-1 sejak awal (`main.go:284`).
- [x] **Step 2.** Regenerated `data/Level_1_Features/modules/broker_aggregate_features.parquet` (252,926 rows × 190 cols, 227 MB). Backup di `*.pre_v21_audit.bak` (45 MB, 48,693 rows — coverage lama lebih sempit). Shift verified: SQ+AADI date 2026-04-20 di L1 same-day = 1.408e9, di feat_agg date=2026-04-20 `sq_flow_total_net_buy` = −5.025e9 (match SQ value pada 2026-04-17, trading day sebelumnya).
- [x] **Step 3.** Regenerated L2 training datamart `data/Level_2_Datamart/training_datamart_bsjp_close10.parquet` (lean: 260 cols × 18,957 rows, 11.5 MB; modules JOINed at train time). Date 2024-11-05 → 2026-05-06, tp_rate 39.04%. Backup `*.pre_v21_audit.bak` (227 MB). Semua 8 module parquet ikut diregenerate.
- [x] **Step 4.** Run `train_lightgbm.py --output-dir model/BSJP/v21_clean_broker --feature-modules-dir … --tp-pct 0.01 --sl-pct -0.02 --oot-valid-days 100` — same hyperparam as v19d.
  - *Result: **FAIL:NEGATIVE_OOT_EXPECTANCY**. OOT AUC dropped from 0.541 (v19d) to **0.508** (v21).*
  - *Observation: Overfit gap widened significantly (0.222). Model probabilities are flat (median 0.21, max 0.27) despite OOT positive rate of 39%.*
  - *Conclusion: **Confirmed smoking gun**. v19d alpha was almost entirely driven by lookahead bias in broker features (Training D vs Inference T-1).*
- [ ] **Step 5.** Compare OOT 100D / 50D / 20D / MaxDD: v21 vs v19d. Update tabel di program.md.
- [ ] **Step 6.** Implement v20 ARA policy on top of v21 (sama veto rules) → `v21+v20`. Walk-forward validation per pre-OOT folds (4 folds yang disinggung program.md:344).
- [ ] **Step 7.** Tambah variant `v21+v20` ke Go predict path & cron. **Jangan replace v19d dulu** — log paralel di `picks_log` selama 30+ trading days untuk live comparison.
- [ ] **Step 8.** Decide promote/abandon based on: (a) honest OOT angka, (b) live PnL 30 hari, (c) operasional fillability.

## Invariants (jangan dilanggar)

- **Cron production tetap pakai v19d** sampai v21 lulus 30+ hari live.
- OOT (last 100 trading days) dipakai SEKALI untuk evaluasi final v21. Tidak boleh dipakai untuk pilih hyperparam atau policy.
- v20 ARA policy yang ditambahkan pada v21 **tidak boleh di-tune** ulang menggunakan OOT — pakai aturan veto yang sama dengan v19d+v20.

## Progress Log

### 2026-05-07 23:xx — Setup
- ✅ Audit lookahead identified the issue (broker family un-shifted in training).
- ✅ Git commit format box-table baseline (`0c3bf30`).
- ✅ Created `model/BSJP/v21_clean_broker/PROGRESS.md` (this file).

### 2026-05-07 23:xx — Step 1: T-1 broker shift patch
- ✅ Edited `edges/bpjs_opening_tp3/scripts/generate_datamart.py:476-505` — added in-place shift block + z/velocity recomputation di awal `build_feature_aggregate`.
- ✅ Smoke test: row date=2026-04-18 dengan SQ same-day=5000 sekarang menghasilkan `sq_flow_total_net_buy=4000` (T-1 value) ✓; aggregate `flow_total_net_buy_sum=4000` ✓; `sq_tfl_net_buy_z_20` direcompute dari `(lag1 - ma_20)/std_20` clipped ±10 ✓; `sq_tfl_net_buy_velocity_20=lag1/ma_20` ✓.

### 2026-05-07 23:55 — Step 2: Regenerate broker module
- ✅ Backup dibuat: `data/Level_1_Features/modules/broker_aggregate_features.parquet.pre_v21_audit.bak` (45 MB).
- ✅ Loaded L1 broksum_datamart (2,928,061 rows, dates 2024-10-01..2026-04-23).
- ✅ Loaded master_broker (61 local-fund brokers, 90 bandar brokers).
- ✅ Called patched `build_feature_aggregate()` → output 252,926 rows × 190 cols.
- ✅ Real-data shift verification: SQ+AADI L1 same-day=1.408e9 pada 2026-04-20 → feat_agg sq_flow_total_net_buy pada 2026-04-20 = −5.025e9 (= SQ value 2026-04-17). PERFECT shift.
- ✅ Wrote `broker_aggregate_features.parquet` (227 MB, 252K rows × 190 cols, date coverage 2024-10-01..2026-04-23, 1910 unique tickers, no NaN in sample column).

### 2026-05-08 00:04 — Step 3: Regenerate L2 training datamart (in progress)
- ✅ Backup dibuat: `data/Level_2_Datamart/training_datamart_bsjp_close10.parquet.pre_v21_audit.bak` (227 MB).
- ⏳ Running: `cd edges/bsjp_overnight_sl2/scripts && python generate_datamart.py --exit-hour 10` di background (task `bklp9znur`).
- 📝 Log: `_LOG/v21_regen_l2_20260508_000415.log`.
- 📝 Monitor `bp71m2lmv` armed untuk ping setiap milestone (Loaded/BrokerAgg/CVD/Modules/Universe/Train written) atau error.
- ⏳ Next setelah selesai: Step 4 — train_lightgbm.py dengan output-dir `model/BSJP/bsjp_v21_clean_broker`.
