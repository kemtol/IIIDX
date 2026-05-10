# BSJP/BPJS Operational Gold Standard Checklist

Dokumen ini adalah daftar kritik skeptis berbasis retrospective yang WAJIB dijawab dengan bukti teknis sebelum model riset dipromosikan ke **Production Inference**.

Checklist ini bukan formalitas. Jika satu poin dijawab dengan "asumsi", "seharusnya", atau "belum dicek", promosi model ditunda. Model boleh tetap menjadi kandidat riset, tetapi tidak boleh disebut production-ready.

## Tujuan

Checklist ini dibuat untuk mencegah pengulangan pola kegagalan yang sudah pernah terjadi:

- headline OOT return terlihat bagus, tetapi kemudian ditemukan leakage atau objective mismatch
- fitur intraday memakai candle yang belum tersedia pada waktu keputusan
- fitur harian/broker/CVD hari T salah ditempel ke row prediksi T, padahal baru aman sebagai T-1
- model riset jalan di Python/L1/L2, tetapi inference harian Go/DuckDB belum mendukung feature set yang sama
- return paper tinggi karena mengabaikan ARA fillability, spread, slippage, atau liquidity
- OOT dipakai berkali-kali sebagai kriteria seleksi sehingga tidak lagi steril

## Scope

Checklist ini berlaku untuk semua kandidat BSJP/BPJS yang akan naik dari riset ke inference, terutama:

- model baru di `model/BSJP/vX/`
- perubahan objective, label, execution rule, atau policy
- perubahan feature module di `data/Level_1_Features/modules/`
- perubahan inference variant, Go feature parity, DuckDB schema, atau cron

## Definisi Temporal

Untuk BSJP `close10` saat ini:

| Konsep | Definisi | Boleh Jadi Feature? |
|---|---|---|
| Decision time | sebelum entry close 15:xx hari T | N/A |
| Entry | close 15:xx hari T | boleh sebagai entry/execution reference, bukan predictive feature |
| Exit | open/price 10:xx hari T+1 | label/backtest only, tidak boleh jadi feature |
| `pre14` | intraday hari T sampai maksimum 14:xx | boleh |
| candle 15:xx hari T | candle entry/closing window | tidak boleh masuk feature `pre14` |
| daily OHLCV hari T | baru final setelah market close | tidak boleh untuk keputusan preclose; harus T-1 atau lebih lama |
| broker flow hari T | finalnya baru aman setelah sumber selesai dan stabil | default harus T-1 kecuali latency live dibuktikan |
| CVD/broker aggregate T | turunan broker flow hari T | default harus T-1 |

Rule utama: row prediksi tanggal T hanya boleh melihat intraday T sampai cutoff valid dan data harian/broker/context yang sudah known dari T-1 atau lebih lama.

## Evidence Standard

Setiap gate harus punya bukti yang bisa dicek ulang:

- path artifact, misalnya `_LOG/...json`, `_LOG/...csv`, `metrics.json`, atau script audit
- command yang dipakai untuk membuat artifact
- tanggal data, window OOT, dan model directory yang diuji
- hasil numerik PASS/FAIL, bukan deskripsi umum
- git hash atau catatan perubahan jika ada patch feature/inference

Status yang boleh dipakai:

| Status | Arti |
|---|---|
| `VERIFIED` | Ada artifact dan hasilnya lulus |
| `FAILED` | Ada artifact dan hasilnya gagal |
| `BLOCKED` | Belum bisa diuji karena data/infrastruktur belum siap |
| `NOT_APPLICABLE` | Tidak relevan untuk model ini, dengan alasan eksplisit |

Status `ASSUMED` tidak diperbolehkan untuk promosi.

---

## GATE 1: Operational Feasibility

Kritik: "Riset sering memakai data yang di dunia nyata belum tersedia saat keputusan dibuat."

### 1.1 Data Latency

Pertanyaan kritik:

- Apakah `yfinance_1h`, `IPOT Broksum`, dan sumber L0 lain yang dipakai model tersedia stabil sebelum decision deadline?
- Apakah latency sudah diukur dari run harian nyata, bukan dari backfill?
- Apakah ada tanggal ketika data telat, kosong, duplicate, atau partial?

Bukti minimum:

- log fetch harian dari `_LOG/` untuk beberapa hari live/paper
- timestamp availability per source
- coverage table per date/ticker/hour
- explicit decision deadline, misalnya sebelum 15:42 WIB untuk preclose signal

Fail condition:

- data hanya tersedia setelah deadline
- data tersedia tetapi coverage tidak konsisten
- belum ada bukti latency live

### 1.2 Inference Parity

Pertanyaan kritik:

- Apakah Go/DuckDB inference sudah mendukung semua fitur model yang dilatih?
- Apakah jumlah, nama, urutan, dtype, dan NaN handling feature sama dengan training?
- Apakah inference tidak membutuhkan rebuild L1/L2 parquet saat cron?

Bukti minimum:

- output calibration Go vs Python
- feature parity report: `PERFECT`, epsilon, missing, extra
- model variant config yang menunjuk ke model file dan feature list yang benar
- binary build log

Fail condition:

- ada feature training yang belum dihitung inference
- inference membaca L1/L2 langsung untuk daily production
- Go output belum dikalibrasi terhadap Python untuk feature baru

### 1.3 No-Trade Safety

Pertanyaan kritik:

- Jika L0 hari ini korup, stale, partial, atau source down, apakah sistem memilih no-trade?
- Apakah no-trade dikirim ke Telegram/Discord/log dengan alasan jelas?
- Apakah sistem mencegah fallback diam-diam ke data kemarin sebagai sinyal hari ini?

Bukti minimum:

- `bsjp check --verbose` atau equivalent preflight
- simulated missing-source test
- log no-trade dengan reason code

Fail condition:

- sistem tetap mengeluarkan picks saat source critical gagal
- stale date tidak terdeteksi
- no-trade hanya manual judgment

### 1.4 Runtime & Deployment

Pertanyaan kritik:

- Apakah full fetch/check/predict selesai dalam SLA VPS?
- Apakah memory dan disk cukup di VPS target?
- Apakah binary Go dan cron tidak bergantung pada environment research Python?

Bukti minimum:

- wall-clock runtime dari clean run
- memory/disk footprint
- cron command final
- successful run dari environment production-like

Fail condition:

- runtime tidak memenuhi SLA
- butuh manual Python install untuk inference harian
- command production belum deterministik

### 1.5 Cost Awareness

Pertanyaan kritik:

- Apakah model/policy memakai estimasi spread dan market impact, bukan hanya fee broker?
- Apakah cost cap dipakai pada ranking atau policy filter?
- Apakah cost proxy dihitung dari data yang tersedia sebelum entry?

Bukti minimum:

- definisi `pre14_spread_cost_est` / `pre14_market_cost_est` atau cost feature set lain
- policy simulation dengan cost cap
- sensitivity table terhadap cost assumption

Fail condition:

- return hanya gross atau hanya fee commission
- spread/slippage tidak diuji
- cost feature memakai data setelah cutoff

---

## GATE 2: Anti-Lookahead Audit

Kritik: "Model ML sangat pintar mencari celah untuk melihat masa depan."

### 2.1 Feature Blacklist & Outcome Leakage

Pertanyaan kritik:

- Apakah `exit_price`, realized return, target label, dan outcome-like columns diblokir dari feature set?
- Apakah feature importance tidak menunjukkan kolom yang secara semantik adalah label tersamar?

Bukti minimum:

- final feature list
- blacklist check artifact
- top feature importance review

Fail condition:

- outcome-like column muncul di model features
- top gain didominasi feature yang secara temporal mustahil

### 2.2 Intraday Cutoff: 15:xx Candle Issue

Pertanyaan kritik:

- Untuk `pre14`, apakah sudah dibuktikan secara fisik tidak ada candle 15:xx yang terpilih?
- Apakah candle 15:xx hanya dipakai sebagai entry/execution reference, bukan predictive feature?
- Apakah raw data memang memiliki candle setelah 14:xx, dan audit membuktikan candle itu tidak dipilih?

Bukti minimum:

- audit reconstruction dari raw `yfinance_1h`
- selected max hour <= 14
- count selected after 14 = 0
- count raw/session bars after 14 > 0 sebagai kontrol negatif

Fail condition:

- ada selected bar hour 15 atau lebih
- close/high/low/volume candle 15:xx masuk ke feature `pre14`
- cutoff hanya dipercaya dari nama kolom tanpa rebuild audit

### 2.3 T-1/T Alignment

Pertanyaan kritik:

- Apakah semua fitur harian yang baru final setelah market close T digeser ke T-1 untuk row T?
- Apakah broker aggregate, CVD, `yf_daily_*`, daily range, high/low/close, context, dan timeflow tidak memakai nilai final T?
- Apakah z-score/velocity dihitung ulang dari shifted values, bukan shift hasil akhir yang sudah tercampur T?

Bukti minimum:

- exact-match audit terhadap fresh shifted rebuild
- same-day match count harus 0 atau dijelaskan untuk feature yang memang intraday-safe
- shifted T-1 match count harus penuh pada comparable rows

Fail condition:

- feature row T match dengan final daily/broker value T
- aggregate dihitung sebelum shift sehingga T bocor lewat mean/z/context
- patch hanya shift beberapa kolom, tetapi turunannya masih memakai T

### 2.4 Broker & CVD Shift Verification

Pertanyaan kritik:

- Apakah broker features menggunakan `groupby(ticker, broker).shift(1)` atau equivalent temporal-safe logic?
- Apakah cross-sectional broker families (`sq_`, `xc_`, `yp_`, `pd_`) sudah diaudit, bukan diasumsikan?
- Apakah CVD rolling memakai flow sampai T-1, bukan termasuk T?

Bukti minimum:

- exact match audit untuk representative broker columns
- CVD T-1 rolling rebuild report
- cross-sectional family audit report jika keluarga feature dipakai model

Fail condition:

- broker/CVD feature T match same-day source
- ada keluarga broker-derived feature yang belum diaudit tetapi dipakai kandidat production

### 2.5 Macro / Global / Daily Context Timing

Pertanyaan kritik:

- Apakah global index, FX, macro, dan daily context memakai kalender yang benar?
- Apakah timezone dan market close source asing tidak membuat value T sebenarnya baru diketahui setelah decision time?

Bukti minimum:

- source availability assumption tertulis
- shift policy per source
- audit sample untuk tanggal lintas timezone

Fail condition:

- global/macro T dipakai tanpa bukti availability sebelum decision time
- timezone tidak eksplisit

### 2.6 Isolation Sandbox Protocol

Pertanyaan kritik:

- Jika file masa depan setelah tanggal cutoff dihapus secara fisik, apakah feature untuk tanggal cutoff tetap identik?
- Apakah model scoring untuk tanggal T bisa direbuild tanpa data T+1 dan masa depan?

Bukti minimum:

- script isolation rebuild
- hash/row-level comparison before vs sandbox
- tested dates mencakup OOT dan recent extension

Fail condition:

- feature berubah ketika future data dihapus
- sandbox belum pernah dijalankan untuk kandidat promosi

### 2.7 Target Integrity

Pertanyaan kritik:

- Apakah `entry_price` benar-benar harga yang tersedia saat keputusan?
- Apakah `exit_price` hanya label/evaluation dan tidak pernah masuk feature?
- Apakah objective tertulis sesuai artifact: `overnight` vs `close10` tidak tertukar?

Bukti minimum:

- label construction audit
- metadata di `metrics.json`: objective, entry time, exit time, label name
- sample rows manual verification

Fail condition:

- artifact bernama satu objective tetapi labelnya objective lain
- exit/return masuk feature
- entry memakai harga yang belum tersedia saat order

---

## GATE 3: Statistical Skepticism

Kritik: "Return tinggi biasanya keberuntungan, overfitting, atau leakage yang belum ketahuan."

### 3.1 Too Good To Be True

Pertanyaan kritik:

- Jika return > 1% per hari secara konsisten, apakah leakage tambahan sudah dicari?
- Apakah AUC berada di zona realistis untuk financial ML?
- Apakah return tinggi berasal dari volatility compounding, bukan edge yang stabil?

Bukti minimum:

- AUC, AUCPR, precision@k, mean/day, vol/day, max drawdown
- leakage audit artifact
- top feature review

Fail condition:

- AUC OOT > 0.65 tanpa audit sangat kuat
- return sangat tinggi tetapi AUC hampir random dan tidak ada diagnostics
- best feature gain terlalu dominan tanpa penjelasan

### 3.2 OOT Sterility

Pertanyaan kritik:

- Apakah OOT hanya dipakai sekali untuk final evaluation?
- Apakah hyperparameter, policy threshold, q-cut, k, max weight, dan cost cap tidak dipilih dengan mengejar OOT final?

Bukti minimum:

- run selection log
- pemisahan train/valid/OOT
- catatan apakah policy dipilih dari validation atau diagnostic OOT

Fail condition:

- model/policy dipilih dari best OOT setelah banyak percobaan lalu disebut final
- tidak ada audit selection bias

### 3.3 Rolling Stability

Pertanyaan kritik:

- Apakah performa positif di beberapa jendela waktu, bukan hanya 100 hari terakhir?
- Apakah fold AUC dan portfolio metrics stabil?
- Apakah latest extension dipisahkan dari locked OOT?

Bukti minimum:

- rolling-retrain atau walk-forward validation
- per-fold AUC/return/drawdown
- recent 7D/30D/90D marked as provisional jika bukan locked OOT

Fail condition:

- hanya punya 100D OOT tunggal
- latest extension dicampur dengan OOT locked
- satu fold collapse tanpa diagnosis

### 3.4 Ablation Check

Pertanyaan kritik:

- Jika family feature paling kuat dihilangkan, apakah model masih punya edge?
- Apakah broker, pre14, macro, ARA-history, dan CVD contribution dipahami?

Bukti minimum:

- no-broker / no-pre14 / no-ARA-history atau ablation yang relevan
- feature-family gain attribution
- per-family performance comparison

Fail condition:

- seluruh edge hilang saat satu feature-family dicabut dan tidak ada alasan production-safe
- family yang belum diaudit menyumbang gain material

### 3.5 Concentration Risk

Pertanyaan kritik:

- Apakah profit hanya berasal dari 1-2 ticker, satu sector, atau satu event regime?
- Apakah top winners/outliers mendominasi equity curve?

Bukti minimum:

- PnL by ticker, sector, liquidity bucket, price bucket
- contribution top 1/5/10 tickers
- worst-day and best-day review

Fail condition:

- mayoritas PnL berasal dari outlier kecil
- model tidak survive ketika top outliers dihapus

### 3.6 Drawdown & Tail Risk

Pertanyaan kritik:

- Apakah max drawdown, worst day, VaR/ES, dan recovery profile sesuai modal target?
- Apakah policy punya exposure cap yang masuk akal?

Bukti minimum:

- daily return distribution
- Monte Carlo/block bootstrap jika kandidat sudah melewati audit awal
- stress test k/weight/q/cost

Fail condition:

- drawdown tidak acceptable untuk ukuran modal
- no exposure cap pada model high-volatility

---

## GATE 4: Fillability & Liquidity

Kritik: "Harga di layar belum tentu harga yang bisa didapat."

### 4.1 ARA/ARB Fillability

Pertanyaan kritik:

- Apakah model menyarankan saham yang sudah terkunci ARA saat entry?
- Jika ARA-like tetap dipilih, apakah ada bukti antrean/fillability realistis?
- Apakah ARA-state digunakan sebagai model feature, policy filter, atau diagnostic saja?

Bukti minimum:

- ARA touch/locked/release diagnostic
- policy simulation dengan ARA veto/filter
- trade examples pada bucket ARA

Fail condition:

- return bergantung pada beli saham locked ARA
- ARA filter dipilih dari OOT tanpa fresh validation lalu dipromosikan

### 4.2 Spread & Slippage

Pertanyaan kritik:

- Apakah expected profit masih positif setelah spread, fee, dan slippage?
- Apakah cost cap cukup konservatif untuk saham lapis 2/3?

Bukti minimum:

- spread/slippage sensitivity
- execution-cost-adjusted return
- cost bucket performance

Fail condition:

- edge hilang pada slippage kecil
- spread estimate memakai candle/quote yang belum tersedia saat decision

### 4.3 Liquidity Capacity

Pertanyaan kritik:

- Apakah volume bid/ask cukup untuk modal target tanpa menggerakkan harga?
- Apakah Rp10m, Rp50m, Rp100m, atau capacity target lain diuji eksplisit?

Bukti minimum:

- turnover/volume bucket diagnostics
- participation rate estimate
- capacity simulation per pick

Fail condition:

- picks dominan di ticker sangat tipis
- modal target tidak bisa masuk tanpa price impact material

### 4.4 Price Tier & Tick Size

Pertanyaan kritik:

- Apakah saham di bawah threshold harga disaring?
- Apakah tick size terlalu besar relatif terhadap target profit?
- Apakah ARB/UMA/suspension-like names punya guard?

Bukti minimum:

- price filter policy
- price bucket performance
- tick-size cost estimate

Fail condition:

- model membeli saham di price tier yang membuat spread/tick menghapus edge
- no guard untuk saham tidak tradable/suspended

---

## GATE 5: Production Handoff

Kritik: "Model clean di riset belum tentu deployable sebagai signal service."

### 5.1 Artifact Completeness

Pertanyaan kritik:

- Apakah model directory lengkap dan reproducible?
- Apakah `metrics.json` mencatat objective, feature count, model path, OOT range, policy, dan checklist status?

Bukti minimum:

- `model_lightgbm_*.txt`
- `metrics.json`
- `feature_importance.csv`
- `valid_predictions.parquet`
- command/run script atau session note

Fail condition:

- model tidak bisa direproduce
- metrics tidak mencatat objective dan policy

### 5.2 Variant Config

Pertanyaan kritik:

- Apakah inference variant menunjuk model, feature list, policy, filters, dan thresholds yang sama dengan riset?
- Apakah tidak ada mismatch `overnight` vs `close10`?

Bukti minimum:

- variant config diff/review
- dry-run predict output
- pick log schema review

Fail condition:

- inference memakai policy/model/objective berbeda dari artifact riset
- nama variant misleading

### 5.3 Monitoring & Alerting

Pertanyaan kritik:

- Apakah production akan mendeteksi stale data, missing feature, zero picks, abnormal picks, dan failed notification?
- Apakah alert membedakan no-trade yang sehat vs error?

Bukti minimum:

- preflight output
- Telegram/Discord test
- daily log example

Fail condition:

- failure diam-diam
- tidak ada audit trail untuk sinyal harian

---

## Promotion Protocol

Sebelum promosi, buat artifact ringkas untuk model kandidat:

```text
model: model/BSJP/<variant>/
objective: close10 | overnight | other
entry: ...
exit: ...
oot_window: ...
policy: ...
checklist_version: model/OPERATIONAL_CHECKLIST.md
gate_1_operational: VERIFIED | FAILED | BLOCKED | NOT_APPLICABLE
gate_2_lookahead: VERIFIED | FAILED | BLOCKED | NOT_APPLICABLE
gate_3_statistics: VERIFIED | FAILED | BLOCKED | NOT_APPLICABLE
gate_4_fillability: VERIFIED | FAILED | BLOCKED | NOT_APPLICABLE
gate_5_handoff: VERIFIED | FAILED | BLOCKED | NOT_APPLICABLE
evidence:
  - _LOG/...
  - model/BSJP/.../metrics.json
  - ...
```

Rules:

1. Semua gate yang relevan harus `VERIFIED`.
2. `BLOCKED` berarti model tetap research-only.
3. `FAILED` berarti model tidak boleh dipromosikan sampai akar masalah diperbaiki dan audit diulang.
4. `NOT_APPLICABLE` harus punya alasan eksplisit.
5. `metrics.json` model yang dipromosikan harus mereferensikan checklist ini dan evidence artifact.
6. `model/BSJP/LATEST.md` harus diperbarui jika status model berubah.
7. `_MEMORY/YYYYMMDDHHMMSS.md` harus mencatat keputusan promosi atau alasan penundaan.

## Fast Rejection Rules

Model langsung research-only jika salah satu kondisi ini terjadi:

- feature `pre14` memilih candle 15:xx atau lebih
- feature harian/broker/CVD row T match dengan final value T tanpa bukti availability sebelum decision time
- outcome/exit/realized return masuk feature set
- OOT dipakai sebagai selection target utama lalu diklaim final
- Go inference belum bisa menghitung feature set model
- production path membutuhkan L1/L2 rebuild harian
- picks bergantung pada saham locked ARA yang tidak fillable
- tidak ada no-trade fallback untuk stale/missing critical source
