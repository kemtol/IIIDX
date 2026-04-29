# PRD 0003 — L0 Migration to DuckDB

| Field | Value |
|---|---|
| Status | Draft |
| Date | 2026-04-29 |
| Author | mkemalw |
| Related | [program.md](../../program.md) §1, [RETROSPECTIVE.md](../../RETROSPECTIVE.md), PRD 0001, PRD 0002 |
| Scope | **L0 only.** L1/L2 working store dan training snapshot adalah PRD terpisah. |

---

## 0) Related PRD & Sequencing

Migrasi storage dari parquet ke DuckDB dijalankan dalam beberapa PRD bertahap:

| PRD | Scope | Status |
|---|---|---|
| **0003 (ini)** | L0 raw data → DuckDB per source | Draft |
| 0004 (future) | L1/L2 working feature store di DuckDB | TBD |
| 0005 (future) | Training snapshot pattern (frozen parquet per model version) | TBD |

PRD ini sengaja sempit: **L0 only**, supaya playbook migrasi divalidasi di surface dengan blast radius terkecil sebelum diterapkan ke layer yang lebih berisiko.

---

## 1) Why Now

### 1.1 Pain points yang dialamat

| Pain point | Source | Frekuensi |
|---|---|---|
| yfinance fetch gagal di tengah jalan, parquet bisa partial-rewrite | `pipeline/fetch/yfinance_*.py` | Sering |
| Resume state JSON manual untuk broksum (`_LOG/broksum_resume_state.json`) | `pipeline/fetch/broksum.py` | Setiap run |
| Single-file parquet rewrite = no atomic update, no row-level upsert | Semua L0 | Setiap fetch |
| Tidak ada audit trail "kapan baris ini masuk, dari fetch run mana" | Semua L0 | Always |
| Disk growth tidak terdedup natural — ticker yang sama berulang | yfinance | Tumbuh seiring waktu |

### 1.2 Foundation untuk PRD berikutnya

Migrasi L0 dulu memberi:
- **Validasi pattern dual-write + canary** sebelum diterapkan ke L1/L2.
- **Operational confidence** dengan DuckDB beyond inference-only.
- **API helper** (`attach_l0`) yang akan dipakai L1/L2 builder nanti.

### 1.3 Yang TIDAK diselesaikan PRD ini

| Issue | Rencana |
|---|---|
| Insiden v18 (rebuild L1/L2 irreversible) | PRD 0004 — L1/L2 working store |
| Training reproducibility (snapshot per model version) | PRD 0005 — Training snapshot |
| Feature R&D agility (incremental upsert kolom baru) | PRD 0004 |

Set ekspektasi: **L0-first migration tidak mencegah v18 berikutnya.** Ini foundation work yang nilainya compound nanti.

---

## 2) Goals & Non-goals

### Goals

1. Stop yfinance partial-fetch corruption via transactional writes (BEGIN/COMMIT/ROLLBACK).
2. Replace JSON resume state dengan SQL query langsung ke DB.
3. Aktifkan natural dedup via `INSERT OR REPLACE` / `MERGE INTO` per natural key.
4. Pertahankan parallel fetch capability (yfinance daily, 1h, 4h jalan barengan).
5. Migrasi bertahap (canary by source) supaya rollback granular per source.
6. Zero downtime untuk inference (read path inference tidak berubah selama migrasi).

### Non-goals

1. ❌ Mengubah feature engineering logic (`generate_broksum_datamart.py`, dst).
2. ❌ Mengganti fetch logic (yfinance API, IPOT websocket, dll).
3. ❌ Menggabung semua L0 ke satu DB file (anti-goal — lihat §4.1).
4. ❌ Menyentuh L1/L2/inference store di PRD ini.

---

## 3) Preconditions (HARD)

| # | Precondition | Reason |
|---|---|---|
| 1 | **`git init` di repo root, commit baseline** sebelum touch any fetch script. | Tanpa version control, modifikasi fetch script tidak bisa di-rollback. Insiden v18 lesson. |
| 2 | Backup parquet L0 saat ini ke `_BAK/L0_pre_duckdb_<date>/` | Last-resort recovery selama migrasi. |
| 3 | Tag baseline state di `_MEMORY/<timestamp>_pre_l0_migration.md` | Snapshot operasional untuk handoff antar session. |
| 4 | **Continuity test** `pipeline/storage/tests/test_continuity.py` sudah ada dan harus pass sebelum merge tiap fetch script refactor. Test assert: untuk tiap source, parquet output post-refactor dengan default flags identik dengan parquet output pre-refactor (byte-equal atau `np.isclose` per dtype rules §5.3). | Menjamin gateway pattern (`write_l0()`) adalah passthrough sempurna saat default flags — current pipeline tidak pernah putus selama migrasi. |

PR pertama di phase manapun **tidak boleh merge** sampai keempat precondition di atas selesai.

### Continuity Invariants (selama migrasi berjalan)

Tiga aturan operasional yang **wajib** dipegang supaya pipeline existing tetap jalan:

1. **Default flag state = current behavior.** Code yang di-merge selalu pakai default `parquet_write=True`, `duckdb_write=False`, `duckdb_read=False`. Promosi stage hanya via env var, bukan code change.
2. **Stage 3 (parquet retire) HANYA setelah read-side migrated.** Sebelum `parquet_write=False`, semua downstream reader untuk source itu wajib sudah pakai `read_l0()`. Audit checklist per source di Phase cutover.
3. **Inference path tidak disentuh.** L1/L2 parquet tetap di-write dari L0 (yang masih dual-write atau parquet-only), `inference.duckdb` tetap dapat data fresh, inference cron jalan normal sepanjang migrasi.

---

## 4) Architecture

### 4.1 Topology — Per-Source Files (Not Single File)

```
data/Level_0_Raw/
├── master.duckdb              # master_emiten + master_broker (low write freq, gabung OK)
├── global_indices.duckdb      # ^IXIC, ^N225, ^VIX, USD/IDR, IHSG, dll
├── yfinance_daily.duckdb      # daily OHLCV per ticker
├── yfinance_1h.duckdb         # intraday 1h
├── yfinance_4h.duckdb         # intraday 4h
└── broksum.duckdb             # broker summary, heaviest, websocket
```

**Alasan separasi**:
- DuckDB pakai file-level lock untuk write. Single file = serialized writes across processes.
- Fetch script `run_fetch_yfinance_daily.sh`, `run_fetch_yfinance.sh` (1h), `run_fetch_yfinance.sh` (4h) saat ini jalan paralel di cron — separasi mempertahankan paralelisme.
- Backup, rollback, dan migrasi canary jadi granular per source.

**Master digabung** karena: write frequency rendah, fetch script tidak overlap, dan L1 sering join master_emiten × master_broker.

### 4.2 Schema per Source

| File | Table | Natural Key (PK) | Notes |
|---|---|---|---|
| `master.duckdb` | `master_emiten` | `ticker` | Static-ish, ~800 rows |
| `master.duckdb` | `master_broker` | `broker_code` | Static, ~92 rows |
| `global_indices.duckdb` | `global_indices` | `(symbol, date)` | Multi-symbol time series |
| `yfinance_daily.duckdb` | `yfinance_daily` | `(ticker, date)` | Daily OHLCV |
| `yfinance_1h.duckdb` | `yfinance_1h` | `(ticker, datetime)` | Intraday |
| `yfinance_4h.duckdb` | `yfinance_4h` | `(ticker, datetime)` | Intraday |
| `broksum.duckdb` | `broksum_bybroker` | `(date, broker, ticker)` | Heaviest table |

**Setiap tabel WAJIB punya kolom audit** (ditulis eksplisit oleh app layer, **tidak** pakai DB trigger):

- `_inserted_at TIMESTAMP` — set sekali saat row pertama di-insert
- `_updated_at TIMESTAMP` — set ulang setiap upsert

Pattern di SQL:

```sql
-- Insert baru
INSERT INTO <table> (<cols>, _inserted_at, _updated_at)
VALUES (<vals>, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);

-- Upsert (update path)
INSERT INTO <table> (<cols>, _inserted_at, _updated_at)
VALUES (<vals>, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT (<natural_key>) DO UPDATE
  SET <col> = EXCLUDED.<col>,
      _updated_at = CURRENT_TIMESTAMP;
-- _inserted_at sengaja TIDAK di-overwrite saat conflict
```

**Alasan no-trigger**: predictability + portability antar versi DuckDB. App-layer = visible di code review, debuggable, tidak bergantung fitur DB yang behavior-nya bisa berubah.

### 4.3 Cross-Source Query via ATTACH

L1 builder dan auditor bisa join lintas-source tanpa gabung file:

```python
con = duckdb.connect()
con.execute("ATTACH 'data/Level_0_Raw/yfinance_daily.duckdb' AS yfd (READ_ONLY)")
con.execute("ATTACH 'data/Level_0_Raw/global_indices.duckdb' AS gi (READ_ONLY)")
con.execute("ATTACH 'data/Level_0_Raw/master.duckdb' AS m (READ_ONLY)")

con.sql("""
  SELECT y.ticker, y.date, y.close, g.value AS vix, m.sector
  FROM yfd.yfinance_daily y
  LEFT JOIN gi.global_indices g ON y.date = g.date AND g.symbol = '^VIX'
  LEFT JOIN m.master_emiten m ON y.ticker = m.ticker
  WHERE y.date >= '2026-01-01'
""").df()
```

DuckDB query planner menangani lintas-file efficiently — performa setara dengan single-file.

### 4.4 Helper Library

Bikin helper module `pipeline/storage/l0.py`:

```python
L0_DIR = Path("data/Level_0_Raw")
L0_SOURCES = [
    "master", "global_indices", "yfinance_daily",
    "yfinance_1h", "yfinance_4h", "broksum",
]

def attach_l0(con: duckdb.DuckDBPyConnection,
              sources: list[str] | None = None,
              read_only: bool = True) -> None:
    """Attach L0 source DBs to a connection for cross-source queries."""
    mode = "(READ_ONLY)" if read_only else ""
    for src in (sources or L0_SOURCES):
        path = L0_DIR / f"{src}.duckdb"
        con.execute(f"ATTACH '{path}' AS {src} {mode}")

def open_l0_writer(source: str) -> duckdb.DuckDBPyConnection:
    """Open a write connection to a single L0 source. Acquires file lock."""
    return duckdb.connect(str(L0_DIR / f"{source}.duckdb"))
```

Semua reader baru gunakan `attach_l0()`. Semua writer gunakan `open_l0_writer()`. Hardcoded path dilarang.

### 4.5 Concurrency Hardening

Tambah flock di setiap fetch shell script untuk cegah double-run:

```bash
# pipeline/run/run_fetch_yfinance_daily.sh
flock -x -w 60 /tmp/idx_fetch_yfinance_daily.lock \
  python pipeline/fetch/yfinance_daily.py
```

Ini orthogonal dengan DuckDB's lock — flock di shell layer mencegah race condition lebih awal dengan error message yang jelas.

---

## 5) Migration Strategy — Canary by Source

### 5.1 Phase Order

| Phase | Source | DuckDB File | Rationale |
|---|---|---|---|
| 1 | master_emiten + master_broker | `master.duckdb` | Smallest, static, validate dual-write pattern |
| 2 | global_indices | `global_indices.duckdb` | Test yfinance flakiness handling |
| 3 | yfinance_daily | `yfinance_daily.duckdb` | Volume sedang, daily cadence |
| 4a | yfinance_1h | `yfinance_1h.duckdb` | Bisa paralel dengan 4b |
| 4b | yfinance_4h | `yfinance_4h.duckdb` | Bisa paralel dengan 4a |
| 5 | broksum_bybroker | `broksum.duckdb` | Heaviest, websocket, resume state |

Setiap phase = independent migration. Pause di phase manapun jika ada masalah.

### 5.2 Dual-Write Pattern (Per Phase)

```
                ┌──> parquet (LEGACY — kept as immutable archive)
fetch script ──┤
                └──> DuckDB (NEW — validated)
                       │
                       └──> daily validator job
                            asserts DB == parquet (count, range, hash sample)
```

**Implementasi minimum di fetch script**:

```python
# Pseudocode untuk fetch_yfinance_daily.py setelah modifikasi
def fetch_and_persist(date_range):
    df = fetch_from_yfinance(date_range)  # unchanged

    # Legacy write (parquet) — tetap sampai cutover
    write_parquet(df, "data/Level_0_Raw/yfinance_daily.parquet")

    # New write (DuckDB) — transactional dengan audit timestamp eksplisit
    with open_l0_writer("yfinance_daily") as con:
        con.execute("BEGIN")
        try:
            con.execute("""
                INSERT INTO yfinance_daily
                SELECT *, CURRENT_TIMESTAMP AS _inserted_at, CURRENT_TIMESTAMP AS _updated_at
                FROM df
                ON CONFLICT (ticker, date) DO UPDATE SET
                    open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                    close = EXCLUDED.close, volume = EXCLUDED.volume,
                    _updated_at = CURRENT_TIMESTAMP
            """)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
```

### 5.3 Validation Criteria (Per Source)

Daily validator job, output ke `_LOG/duckdb_canary_<source>.jsonc`.

**Hard checks (strict equality)**:

| Check | Threshold |
|---|---|
| `count(*)` parquet == DuckDB | Equal |
| `count(distinct natural_key)` match | Equal |
| `min(date), max(date)` match | Equal |
| Schema (column names + dtypes) match | Equal |
| Null count per kolom match | Equal |

**Sample comparison (1000 random rows, sorted by natural key, seed=42)** — toleransi per dtype:

| Dtype | Comparison rule |
|---|---|
| `int*`, `uint*` | Strict `==` |
| String / categorical | Strict `==` |
| Boolean | Strict `==` |
| Datetime / timestamp | Strict `==` setelah normalize ke UTC |
| `float32`, `float64` | `np.isclose(rtol=1e-9, atol=1e-12, equal_nan=True)` |

**Alasan `rtol=1e-9` untuk float**: round-trip pandas ↔ pyarrow ↔ DuckDB typically introduces drift di orde 1e-15 (machine epsilon) sampai 1e-12 (worst case). Real corruption (typo, swap, dtype error, scaling bug) menghasilkan drift ≥ 1e-3 — gap-nya 6+ orders of magnitude. 1e-9 ketat untuk catch real bug, longgar untuk tolerate round-trip noise.

**Pass criteria untuk cutover**: zero discrepancy untuk **14 hari kalender berturut-turut** (mencakup minimal 2 minggu trading + recovery dari weekend/holiday gaps).

### 5.4 Cutover Pattern (Per Source)

Cutover dilakukan dalam 2 sub-phase:

**Sub-phase A: read switch**
- Downstream code (L1 builder, auditor, dll) ganti baca dari DuckDB via `attach_l0()`.
- Fetch script tetap dual-write.
- Run minimal 7 hari untuk validate read path stabil.

**Sub-phase B: write retire**
- Fetch script stop tulis parquet, hanya tulis DuckDB.
- Parquet legacy tetap di disk sebagai archive (tidak dihapus).
- Update `program.md` §3 untuk reflect topologi baru.

### 5.5 Rollback Plan

Per phase, rollback path:

| Stage | Rollback action |
|---|---|
| Pre dual-write | Revert fetch script via git |
| Dual-write | Stop tulis ke DuckDB, downstream tetap baca parquet |
| Read switch | Revert downstream config: `L0_BACKEND=parquet` |
| Write retire | Restore parquet write di fetch script (mtime baru, tapi DuckDB masih punya history) |

Rollback **per source** tidak mempengaruhi source lain karena topologi separated.

### 5.6 Resume State Migration

Saat ini `_LOG/broksum_resume_state.json` melacak progres fetch broksum. Setelah Phase 5:

```sql
-- Replace JSON state with SQL query
SELECT
  ticker,
  MAX(date) AS last_fetched_date,
  COUNT(*) AS row_count
FROM broksum.broksum_bybroker
GROUP BY ticker;
```

JSON state tetap ditulis paralel selama dual-write Phase 5 untuk cross-check.

---

## 6) Operational Considerations

### 6.1 Backup Strategy

- **Per source backup**: cron `EXPORT DATABASE` setiap `*.duckdb` ke `_BAK/duckdb_<source>_<tier>_<date>/` dalam format parquet.
- **Cron job**: `pipeline/run/run_backup_l0_duckdb.sh` (single script, handle ketiga tier via flag).

**Retention tiers**:

| Tier | Frequency | Keep | Storage path pattern |
|---|---|---|---|
| Weekly | Setiap Minggu 02:00 WIB | Last 4 | `_BAK/duckdb_<source>_weekly_<YYYYMMDD>/` |
| Monthly | Minggu pertama tiap bulan | Last 12 | `_BAK/duckdb_<source>_monthly_<YYYYMM>/` |
| Yearly | Minggu pertama bulan Januari | All (archive, never delete) | `_BAK/duckdb_<source>_yearly_<YYYY>/` |

**Cleanup logic**: backup script otomatis hapus weekly/monthly yang sudah lewat retention. Yearly archive **tidak pernah dihapus** (full audit history).

**Disk budget estimate**: per source ~500MB compressed parquet. Total semua tier semua source jangka panjang: ~30GB setelah 5 tahun. Acceptable.

### 6.2 File Size Management

DuckDB tidak punya VACUUM full sekuat Postgres. Mitigasi:
- Run `CHECKPOINT` setelah bulk insert besar (di akhir fetch script).
- Quarterly: `EXPORT DATABASE` → drop file → `IMPORT DATABASE` (full rebuild).

### 6.3 Monitoring

Tambah ke daily cron:
- File size per `*.duckdb` ke metrics log.
- Row count per table.
- Last successful write timestamp.

Output ke `_LOG/duckdb_health_<date>.jsonc`.

### 6.4 Inference Impact

Inference cron (`pipeline/run/run_inference_bsjp.sh`) saat ini **tidak baca L0 langsung** — baca dari `inferences/bsjp/db/inference.duckdb`. Migrasi L0 ini transparent untuk inference.

Caveat: kalau inference suatu saat butuh raw data (debug, recompute feature), pakai `attach_l0()` read-only — tidak akan konflik dengan fetch writer karena read connection tidak block writer.

---

## 7) Implementation Phases (Deliverables)

### Phase 0 — Preconditions

- [ ] `git init` + commit baseline.
- [ ] Backup parquet L0 ke `_BAK/L0_pre_duckdb_<date>/`.
- [ ] Tulis `_MEMORY/<timestamp>_pre_l0_migration.md`.
- [ ] Bikin `pipeline/storage/l0.py` dengan `attach_l0()`, `open_l0_writer()`, `L0_SOURCES`.
- [ ] Bikin `pipeline/storage/config.py` dengan `L0SourceConfig` + `load_config()` (§12.2).
- [ ] Bikin schema registry `pipeline/storage/schemas/_base.py` + template `master_emiten.py` (§12.1).
- [ ] Bikin gateway `pipeline/storage/writers.py` + `pipeline/storage/readers.py` (§12.3, §12.4).
- [ ] Bikin script validator template `pipeline/storage/validators/_base.py` dengan tolerance config (§5.3).
- [ ] **Bikin continuity test** `pipeline/storage/tests/test_continuity.py` (Precondition #4 §3) — test parity parquet pre/post refactor untuk setiap source. **Test ini gate untuk merge tiap fetch script refactor.**
- [ ] Bikin backup cron `pipeline/run/run_backup_l0_duckdb.sh` dengan tier weekly/monthly/yearly (§6.1).
- [ ] **Benchmark Phase 5 prep**: load 1 bulan broksum data ke DuckDB ephemeral via per-week chunks. Ukur wall-time + memory peak. Validate per-week default (§5.6 / Phase 5). Kalau anomaly (> 60s per chunk atau peak memory > 1GB) → adjust ke per-tanggal.
- [ ] Pin DuckDB version di `requirements.txt`.

### Phase 1 — master.duckdb (master_emiten + master_broker)

- [ ] Schema migration script: `pipeline/migrations/0003_phase1_master.py` (parquet → duckdb initial load).
- [ ] Modify `pipeline/fetch/master_emiten.py` dan `pipeline/fetch/master_broker.py` jadi dual-write.
- [ ] Validator config: `pipeline/storage/validators/master.py`.
- [ ] Run dual-write 14 hari kalender.
- [ ] Cutover read: ganti reader di `generate_broksum_datamart.py` (kalau pakai master).
- [ ] Cutover write: stop tulis parquet, archive parquet legacy.
- [ ] Update `program.md` §3 untuk master.

### Phase 2 — global_indices.duckdb

(Pola sama dengan Phase 1, target file `global_indices`.)

### Phase 3 — yfinance_daily.duckdb

(Pola sama, target `yfinance_daily`.)

### Phase 4a/4b — yfinance_1h.duckdb + yfinance_4h.duckdb

(Bisa paralel. Pola sama.)

### Phase 5 — broksum.duckdb

- [ ] Schema migration script `pipeline/migrations/0003_phase5_broksum.py`:
  - Chunked load **per minggu** (~200K rows per transaction, ~104 commit untuk 2 tahun history).
  - Resume state di `_LOG/migration_phase5_progress.json` — JSON `{"last_completed_week": "2024-W42"}`.
  - Per chunk: `BEGIN` → bulk insert → validator (count + sample hash) → `COMMIT` → `CHECKPOINT`.
  - Kalau crash di tengah: resume dari minggu pertama yang belum commit.
- [ ] Dual-write modify `pipeline/fetch/broksum.py`.
- [ ] Migrate resume state JSON → SQL query (sub-phase tersendiri di akhir Phase 5).
- [ ] Run dual-write minimal 14 hari + 1 cycle resume restart untuk validate.
- [ ] Cutover read di `pipeline/feature/generate_broksum_datamart.py`.
- [ ] Cutover write.
- [ ] Update `program.md` §3.

---

## 8) Success Metrics

| Metric | Target | Measurement |
|---|---|---|
| Phase 1 completion | master cutover penuh | Master parquet write retired |
| All phases completion | Semua L0 source di DuckDB | 6 file `*.duckdb` di `data/Level_0_Raw/` |
| Concurrent fetch preserved | yfinance daily/1h/4h paralel sama cepat | Wall-clock time fetch ≤ baseline |
| yfinance corruption events | Zero post-cutover | Audit `_LOG/` 30 hari |
| Disk usage | ≤ 1.5x parquet equivalent | `du -sh data/Level_0_Raw/*.duckdb` vs parquet |
| Read latency cross-source | ≤ baseline parquet | Benchmark sample query |
| L1 build time | Tidak regress > 10% | `time bash pipeline/run/run_feature_l1.sh` |

---

## 9) Resolved Decisions

Semua open question dari draft 0.1 sudah diputuskan (2026-04-29):

| # | Topic | Decision | Reflected in |
|---|---|---|---|
| 1 | `_updated_at` mechanism | **App layer** — explicit `CURRENT_TIMESTAMP` di SQL. No DB trigger. | §4.2, §5.2 |
| 2 | Backup retention | **3 tier**: weekly (last 4) + monthly (last 12) + yearly (archive, never delete) | §6.1 |
| 3 | Validator tolerance | **Per-dtype**: strict untuk int/string/bool/datetime, `np.isclose(rtol=1e-9, atol=1e-12)` untuk float | §5.3 |
| 4 | broksum chunk size | **Per minggu** (~200K rows/transaction, ~104 commit). Phase 0 benchmark validate default. | §7 Phase 0, Phase 5 |
| 5 | Stockbit XL scope | **Out-of-scope** — Stockbit XL adalah L1 module derived dari broksum, masuk PRD 0004 (L1 working store). | §0, §11 |

---

## 10) Risk & Mitigation

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| DuckDB version upgrade breaks file format | Data inaccessible | Low | Pin DuckDB version di `requirements.txt`; test upgrade di staging |
| Validator false positive (parquet vs DuckDB drift karena dtype) | False migration block | Medium | Tolerance config; manual review log mingguan selama Phase 1 |
| Cron overlap antara fetch + backup | Lock contention | Low | Flock + cron schedule offset; backup di luar fetch window |
| Disk pressure dari dual-write | OOD pada VPS/dev machine | Medium | Monitor `_LOG/duckdb_health`; weekly cleanup parquet legacy lama |
| Phase 5 (broksum) gagal di tengah migrasi besar | Resume state inconsistent | Medium | Chunked migration + validator per chunk; bisa resume |

---

## 11) Out of Scope (Future PRDs)

- **PRD 0004**: L1 working feature store di DuckDB — supports incremental upsert kolom baru, schema evolution.
  - **Stockbit XL features** dimasukkan di sini (derived dari broksum, jadi L1 module, bukan L0 source).
  - Modul L1 lain: closing momentum, overnight history, broker aggregate, CVD, global indices features, VWAP.
- **PRD 0005**: Training snapshot pattern — frozen parquet per model version, immutable, reproducible. Mencegah insiden v18-class.
- **PRD 0006** (speculative): Observability layer — query log, fetch SLA dashboard.

---

## 12) Implementation Structure (Code Layout)

Section ini adalah blueprint code untuk implementasi canary. Empat lapis: **storage abstraction**, **schema registry**, **migration runner**, **fetch/reader gateways** dikontrol via feature flag.

### 12.1 Directory Layout

```
pipeline/
├── storage/                          # NEW — abstraction layer
│   ├── __init__.py
│   ├── l0.py                         # attach_l0, open_l0_writer, L0_SOURCES
│   ├── config.py                     # feature flags per source per stage
│   ├── schemas/                      # one schema definition per source
│   │   ├── _base.py                  # TableSchema dataclass
│   │   ├── master_emiten.py
│   │   ├── master_broker.py
│   │   ├── global_indices.py
│   │   ├── yfinance_daily.py
│   │   ├── yfinance_1h.py
│   │   ├── yfinance_4h.py
│   │   └── broksum.py
│   ├── readers.py                    # read_l0(source) — toggle parquet ↔ duckdb
│   ├── writers.py                    # write_l0(source, df) — handles dual-write
│   ├── validators/                   # daily canary validator
│   │   ├── _base.py                  # BaseValidator: hard + sample checks
│   │   └── ...                       # one per source (atau generic via schema)
│   ├── backup.py                     # tier weekly/monthly/yearly logic
│   └── migrate.py                    # migration runner CLI
│
├── migrations/                       # NEW — versioned scripts (idempotent)
│   ├── 0003_phase0_init.py           # CREATE TABLE all sources
│   ├── 0003_phase1_master.py         # parquet → duckdb initial load
│   ├── 0003_phase2_global_indices.py
│   ├── 0003_phase3_yfinance_daily.py
│   ├── 0003_phase4a_yfinance_1h.py
│   ├── 0003_phase4b_yfinance_4h.py
│   └── 0003_phase5_broksum.py        # chunked per-week + resume
│
├── fetch/                            # EXISTING — modify in-place
│   └── *.py                          # call writers.write_l0() instead of direct parquet
│
└── run/
    ├── run_backup_l0_duckdb.sh       # NEW — backup cron
    ├── run_validate_l0_duckdb.sh     # NEW — daily validator cron
    └── run_fetch_*.sh                # EXISTING — tambah flock guard
```

**Prinsip pemisahan**:
- `storage/` adalah library — fetch script tidak tau parquet vs DuckDB, hanya panggil `write_l0(source, df)`.
- `schemas/` single source of truth — satu file per tabel berisi: filename DuckDB, table name, parquet path, natural key, kolom + dtype, value cols (untuk `ON CONFLICT DO UPDATE`).
- `migrations/` versioned dan idempotent — bisa di-rerun aman.
- `validators/` orthogonal — daily cron, tidak nempel ke fetch path.

### 12.2 Feature Flag — Canary State Machine

`pipeline/storage/config.py`:

```python
@dataclass
class L0SourceConfig:
    source: str
    parquet_write: bool = True    # legacy
    duckdb_write: bool = False    # new
    duckdb_read: bool = False     # cutover read switch

def load_config() -> dict[str, L0SourceConfig]:
    """Read from env vars (e.g. L0_MASTER_DUCKDB_WRITE), fallback to default."""
```

Setiap source punya 3 flag → 3 stage progression per phase:

| Stage | parquet_write | duckdb_write | duckdb_read | Min duration |
|---|---|---|---|---|
| **0. Pre-migration** | ✅ | ❌ | ❌ | Default semua source |
| **1. Dual-write** | ✅ | ✅ | ❌ | 14 hari kalender, validator zero-discrepancy |
| **2. Read switch** | ✅ | ✅ | ✅ | 7 hari, downstream parity check |
| **3. Cutover** | ❌ | ✅ | ✅ | Permanent — parquet jadi archive |

**Promosi stage** = ubah env var. **Rollback** = unset env var. Tidak ada code change.

State persist di `_LOG/l0_canary_state.jsonc` (audit trail per perubahan stage):

```jsonc
{
  "2026-04-30T08:00:00Z": {"source": "master", "stage": "1_dual_write", "by": "manual"},
  "2026-05-14T09:30:00Z": {"source": "master", "stage": "2_read_switch", "by": "manual"},
  "2026-05-21T10:00:00Z": {"source": "master", "stage": "3_cutover", "by": "manual"}
}
```

### 12.3 Fetch Script — Gateway Pattern

Semua fetch script jadi seragam:

```python
# pipeline/fetch/master_emiten.py (after refactor)
from pipeline.storage.writers import write_l0
from pipeline.storage.config import load_config

def main():
    df = fetch_master_emiten_from_idx()       # unchanged
    cfg = load_config()["master_emiten"]
    write_l0("master_emiten", df, cfg)         # handles dual-write based on flags

if __name__ == "__main__":
    main()
```

Logic dual-write tersembunyi di `writers.write_l0()`:

```python
# pipeline/storage/writers.py
def write_l0(table: str, df: pd.DataFrame, cfg: L0SourceConfig):
    schema = get_schema(table)
    if cfg.parquet_write:
        _write_parquet(df, schema.parquet_path)        # legacy path
    if cfg.duckdb_write:
        _write_duckdb_upsert(df, schema)                # new path
    # No fallback — kalau dua-duanya False, raise eksplisit (config error)
```

`_write_duckdb_upsert` handle: open writer dengan flock, `BEGIN`, generate `INSERT ... ON CONFLICT DO UPDATE` dari schema definition, set `_inserted_at` / `_updated_at` eksplisit (§4.2), `COMMIT`, `CHECKPOINT`.

### 12.4 Reader — Toggle Pattern untuk Downstream

L1 builder dan auditor pakai shim:

```python
# pipeline/storage/readers.py
def read_l0(source: str, **filters) -> pd.DataFrame:
    cfg = load_config()[source]
    if cfg.duckdb_read:
        return _read_duckdb(source, **filters)
    return _read_parquet(source, **filters)
```

Downstream (e.g. `pipeline/feature/generate_broksum_datamart.py`) cukup ganti:

```python
# Sebelum:
df = pd.read_parquet("data/Level_0_Raw/broksum_bybroker.parquet")
# Sesudah:
df = read_l0("broksum_bybroker", date_from="2024-01-01")
```

Toggle via env var, tidak butuh edit downstream code lagi setelah migrasi tiap source.

### 12.5 End-to-End Flow per Phase (Contoh: Phase 1 Master)

```
Day 0  — Phase 0 prep done. master.duckdb belum ada.
         git init done, parquet backed up, storage/ + schemas/master_*.py ready.

Day 1  — Run migrations/0003_phase1_master.py:
           CREATE TABLE master_emiten + master_broker
           INSERT FROM read_parquet(...) ON CONFLICT DO NOTHING (idempotent initial load)
         Validator manual run: PASS.

Day 1  — Promote master ke stage 1 (dual-write):
           export L0_MASTER_DUCKDB_WRITE=true
         Update _LOG/l0_canary_state.jsonc.
         Cron fetch master jalan setiap hari, dual-write aktif.

Day 1-15 — Daily validator cron:
           pipeline/run/run_validate_l0_duckdb.sh master
           Output: _LOG/duckdb_canary_master_<date>.jsonc
           Target: 14 hari berturut-turut zero discrepancy.

Day 15 — Promote ke stage 2 (read switch):
           export L0_MASTER_DUCKDB_READ=true
         Downstream code (yang panggil read_l0) sekarang baca DuckDB.
         Parquet write masih jalan sebagai safety net.

Day 15-22 — Monitor downstream parity (7 hari clean).

Day 22 — Promote ke stage 3 (cutover):
           export L0_MASTER_PARQUET_WRITE=false
         Parquet write retired. Parquet existing tetap di disk sebagai archive.

Day 22+ — Update program.md §3, _MEMORY/<timestamp>.md note. Lanjut Phase 2.
```

### 12.6 Rollback Shape per Stage

| Stage saat rollback | Action |
|---|---|
| Stage 1 (dual-write) | `unset L0_<source>_DUCKDB_WRITE`. Fetch kembali parquet-only. DuckDB file tetap di disk, tidak dihapus. |
| Stage 2 (read switch) | `unset L0_<source>_DUCKDB_READ`. Downstream kembali baca parquet. |
| Stage 3 (cutover) | `export L0_<source>_PARQUET_WRITE=true`. Restart parquet write. Kalau parquet butuh sync data dari DuckDB, export manual: `COPY (SELECT ...) TO 'parquet' (FORMAT PARQUET)`. |

Rollback **per source** tidak mempengaruhi source lain karena flag di-prefix per source.

### 12.7 Phase 0 Deliverables (Code Skeleton)

| File | Tujuan | Estimasi LoC |
|---|---|---|
| `pipeline/storage/__init__.py` | Package marker | < 5 |
| `pipeline/storage/l0.py` | `attach_l0`, `open_l0_writer`, `L0_SOURCES` | ~50 |
| `pipeline/storage/config.py` | `L0SourceConfig`, `load_config()` | ~60 |
| `pipeline/storage/schemas/_base.py` | `TableSchema` dataclass + helpers | ~80 |
| `pipeline/storage/schemas/master_emiten.py` | First schema as template | ~40 |
| `pipeline/storage/writers.py` | `write_l0()` dengan dual-write logic | ~120 |
| `pipeline/storage/readers.py` | `read_l0()` dengan toggle | ~60 |
| `pipeline/storage/validators/_base.py` | `BaseValidator` hard + sample checks | ~150 |
| `pipeline/storage/backup.py` | Tier backup logic | ~100 |
| `pipeline/storage/migrate.py` | Migration runner CLI | ~50 |
| `pipeline/migrations/0003_phase0_init.py` | CREATE TABLE all sources (idempotent) | ~80 |
| `pipeline/run/run_validate_l0_duckdb.sh` | Daily validator cron | ~20 |
| `pipeline/run/run_backup_l0_duckdb.sh` | Backup cron (tier-aware) | ~30 |

Total: ~13 file baru, ~850 LoC.

### 12.8 Out-of-Scope di §12

- Implementation per-source schema selain `master_emiten` template (akan ditulis per phase saat phase tersebut dikerjakan).
- Implementation `pipeline/migrations/0003_phase{1..5}_*.py` (per phase).
- Modifikasi fetch scripts existing (per phase).
- Modifikasi downstream readers (`generate_broksum_datamart.py`, dll) — terjadi di stage 2 promotion per source.

---

## Revision History

| Date | Version | Changes |
|---|---|---|
| 2026-04-29 | 0.1 | Initial draft. Topologi per-source files, canary by source, dual-write pattern, 6 phases, validator + rollback plan. Following retrospective dan diskusi DuckDB sebagai working store. |
| 2026-04-29 | 0.2 | Resolve 5 open questions: (1) audit timestamps via app layer, no triggers; (2) backup retention 3-tier weekly/monthly/yearly; (3) validator per-dtype tolerance dengan rtol=1e-9 untuk float; (4) broksum chunk per-minggu dengan resume state JSON; (5) Stockbit XL out-of-scope, masuk PRD 0004 (L1 derived dari broksum). Update §4.2, §5.2, §5.3, §6.1, §7, §9, §11. |
| 2026-04-29 | 0.3 | Add §12 Implementation Structure: directory layout, feature flag state machine (3 stages × 3 flags per source), gateway pattern fetch + reader, end-to-end flow per phase, rollback shape per stage, Phase 0 deliverables list (~13 files, ~850 LoC). Blueprint untuk start implementation. |
| 2026-04-29 | 0.4 | Add Precondition #4: continuity test wajib pass sebelum merge tiap fetch script refactor — assert parquet output identik pre/post refactor dengan default flags. Add Continuity Invariants section (§3) — 3 aturan operasional yang menjamin current pipeline tidak putus selama migrasi. Update Phase 0 deliverables (§7) untuk include continuity test + writers/readers/config gateway files. |
