# 0001 — TDD Phase: MVP (Go)

| | |
|---|---|
| **Parent** | `../PRD.md` |
| **Lang** | Go 1.26 |
| **Goal** | End-to-end pipeline (L0 → DuckDB → output) working in < 2 days |
| **Principle** | Setiap story independen |
| **Last verified** | 2026-04-28 18:26 WIB |

### Progress

| Story | Status | Verified |
|---|---|---|
| Story 0 — Skeleton | ✅ done | `./bsjp check` → paths resolved |
| Story 1 — Bootstrap | ✅ done | 108,698 rows from L2 parquet in 0.65s |
| Story 2 — Fetch (momentum) | ✅ done | 753 tickers, 8.0s |
| Story 3 — Predict | ✅ done | Top-3: DCII, UNTR, ITMG |
| Story 4 — Broker features | ✅ done | 26 cols, 27s |
| Story 5 — All remaining modules | ✅ done | +119 cols (yf_daily 58 + global 15) |
| Story 6 — LightGBM inference | ✅ done | 27 trees, 0.38s predict |

---

## ✅ Story 0: Skeleton — CLI + Config + Path Resolution

**As a** developer,
**I want** a binary that can resolve all repo paths and parse CLI args.

### Acceptance

| Given | When | Then |
|---|---|---|
| Binary runs without args | `./bsjp` | Prints `--help` with 4 subcommands |
| Path resolution | `./bsjp check` | Prints resolved paths |
| DuckDB missing | `./bsjp check` | Does NOT crash |

### Implementation

- stdlib `flag` (no external CLI lib needed for MVP)
- `internal/config/config.go`: `FromCWD()` walks up from CWD to find `data/Level_0_Raw/`

**Verified:** `./bsjp` prints subcommands. `./bsjp check` resolves all paths. Build: 5s.

---

## ✅ Story 1: DuckDB Bootstrap + Schema

**As a** system,
**I want** to backfill DuckDB from L2 parquet.

### Acceptance

| Given | When | Then |
|---|---|---|
| L2 parquet exists | `./bsjp bootstrap` | Row count == L2 parquet row count |
| L2 parquet missing | `./bsjp bootstrap` | Exit code 1 |

### Implementation

- `internal/db/db.go`: `Open()`, `InitMVPSchema()`
- `internal/db/bootstrap.go`: Uses DuckDB native `read_parquet()` SQL
- `go-duckdb` (CGO, links prebuilt `libduckdb.so`)
- CGO flags: `CGO_LDFLAGS="-L/usr/local/lib -lduckdb" CGO_CFLAGS="-I/usr/local/include"`

**Verified:** `./bsjp bootstrap` → 108,698 rows in 0.65s. DuckDB native parquet reader, no polars needed.

---

## ✅ Story 2: Fetch — Closing Momentum (1 module)

**As a** system,
**I want** to compute 3 momentum features from yfinance_1h and upsert to DuckDB.

### Acceptance

| Given | When | Then |
|---|---|---|
| `yfinance_1h.parquet` has data for target_date | `./bsjp fetch --date X --force` | Features upserted |
| Already up to date | `./bsjp fetch --date X` | "Already up to date" → exit 0 |

### Feature Spec

```
close_ret_last1h  = (close@15 - close@14) / close@14
close_vs_open_day = (close@15 - open@09)  / open@09
close_range_pct   = (day_high - day_low)  / open@09
entry_price       = close@15
```

### Implementation

- `internal/features/momentum.go`: `parquet.ReadFile[yfBar]()` deserializes 2.7M rows into typed structs
- Per-ticker: collect hourly bars, compute 3 formulas, return `[]Row`
- `db.UpsertDate()`: DELETE + INSERT for target date
- `time.Time` for datetime column (parquet TIMESTAMP_MILLIS → Go)

**Verified:** `./bsjp fetch --date 2026-04-23 --force` → 753 tickers in 8.0s.

---

## ✅ Story 3: Predict — Top-3 by Entry Price (sort-only)

**As a** system,
**I want** to query tickers, sort by entry_price, and log top-3 to picks_log.

### Acceptance

| Given | When | Then |
|---|---|---|
| Features exist for date | `./bsjp predict --date X --log` | Top-3 printed + logged |
| Features empty | `./bsjp predict` | "Run fetch first" → exit 1 |

### Implementation

- `SELECT ticker, entry_price FROM features_store WHERE date = ?`
- `sort.Slice` by entry_price descending
- `db.LogPicks()`: ON CONFLICT upsert

**Verified:** `./bsjp predict --date 2026-04-23 --log` → DCII, UNTR, ITMG. 0.15s.

---

## Story 4: Fetch — Broker Base Features (flow + context)

**As a** system,
**I want** to compute broker flow + context features from `broksum_bybroker.parquet`.

⬜ pending — TBD

---

## Story 5: Fetch — All Remaining Modules (parallel)

⬜ pending — TBD (sub-stories 5a through 5i, match Rust TDD)

---

## Story 6: LightGBM Native Tree-Walk

⬜ pending — TBD

---

## Build Instructions

```bash
cd inferences/bsjp/golang
export CGO_LDFLAGS="-L/usr/local/lib -lduckdb"
export CGO_CFLAGS="-I/usr/local/include"
go build -o bsjp ./cmd/bsjp/
```

## Test Harness

```bash
go test ./...                # unit tests
./bsjp check                 # freshness
./bsjp bootstrap             # backfill
./bsjp fetch --date X --force # compute
./bsjp predict --date X --log # score
```
