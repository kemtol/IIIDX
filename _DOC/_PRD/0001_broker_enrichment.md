# PRD: Broker Enrichment (Layer 2 Baseline)

## 0) Related PRD

- Dokumen induk: [PRD 0000 — BPJS Screener](./0000_bpjs_screener.md)
- Posisi di pipeline induk: `Layer 2: Baseline` pada section `3) How?`
- Dokumen ini adalah spesifikasi teknis detail untuk fase `Data Warehouse & Baseline Enrichment`

## 1) What?

Membangun baseline historis aktivitas broker untuk setiap baris `(date, broker, ticker)` di `master_broker.parquet` agar tahap berikutnya bisa menghitung `velocity`, `specificity`, dan `influence` tanpa look-ahead bias.

### Objective

- Menghasilkan `master_broker_enriched.parquet` dengan 8 kolom MA baseline.
- Menjaga row-level grain tetap sama: `1 row = 1 date + 1 broker + 1 ticker`.
- Menjamin baseline memakai data `H-1` ke belakang saja.

### In Scope

- Data sufficiency check untuk kebutuhan MA-240.
- Backfill historis dari IPOT hanya untuk tanggal yang belum ada.
- Perhitungan MA broker-level dan broker-ticker-level.
- Validasi output dan pelaporan NaN distribution.

### Out of Scope

- Perhitungan feature model (velocity score final, ranking model).
- Labeling target return.
- Training/inference model.

## 2) Why?

Layer ini adalah fondasi statistik untuk menjawab pertanyaan: "aktivitas broker hari ini normal atau anomali?". Tanpa baseline yang bersih, fitur downstream akan bias dan model ranking jadi tidak stabil.

## 3) How?

### Enrichment Scheme

```mermaid
flowchart TD
    subgraph A[Stage A: Data Sufficiency & Backfill]
        direction LR
        A1[Load master_broker.parquet]
        A2[Check trading-day coverage]
        A3[Read parquet cache first]
        A4[Fetch IPOT for missing dates only]
        A5[Append + dedupe + sort]
        A1 --> A2 --> A3 --> A4 --> A5
    end

    subgraph B[Stage B: Baseline Computation]
        direction LR
        B1[Aggregate broker daily total]
        B2[Shift(1) + Rolling MA 20/60/120/240]
        B3[Rolling MA per broker-ticker]
        B4[Join MA columns + validate]
        B1 --> B2 --> B3 --> B4
    end

    A --> B
```

### Functional Requirements

1. **No look-ahead bias**: seluruh MA dihitung dari data sampai `H-1`, bukan termasuk hari `H`.
2. **Minimum history strict**: jika history belum cukup untuk window tertentu, hasil MA harus `NaN`.
3. **Broker-level MA (2-step)**:
- agregasi dulu ke `(date, broker)` dengan sum `total_net_buy`
- rolling MA per broker
- join balik ke grain awal
4. **Broker-ticker MA (direct)**: rolling MA langsung pada grup `(broker, ticker)`.
5. **Window standar**: `20, 60, 120, 240` trading days.
6. **Trading day only**: tidak melakukan imputasi weekend.
7. **Local-first backfill**: cek parquet store dulu, fetch IPOT hanya untuk gap tanggal yang missing.
8. **Graceful degradation**: jika fetch gagal/timeout, proses enrichment tetap lanjut dan kasih warning NaN elevated.

### Data Flow Rules

- Kunci deduplikasi: `(date, broker, ticker)`.
- Kolom original tidak boleh berubah nama/semantik.
- Urutan output disarankan sort by `date, broker, ticker` untuk konsistensi.

## 4) Input

### Base File

`master_broker.parquet`

### Required Columns

| Column | Type | Notes |
|---|---|---|
| `date` | date/string | trading date |
| `broker` | string | broker code |
| `ticker` | string | emiten code |
| `total_net_buy` | numeric | net buy value |

## 5) Output

### Output File

`master_broker_enriched.parquet`

### Added Columns

| Column |
|---|
| `ma_total_netbuy_20` |
| `ma_total_netbuy_60` |
| `ma_total_netbuy_120` |
| `ma_total_netbuy_240` |
| `ma_ticker_netbuy_20` |
| `ma_ticker_netbuy_60` |
| `ma_ticker_netbuy_120` |
| `ma_ticker_netbuy_240` |

## 6) Acceptance Criteria

- Row count output identik dengan input.
- Tidak ada `NaN` pada kolom original (`date`, `broker`, `ticker`, `total_net_buy`).
- `NaN` hanya ada pada kolom MA.
- Persentase `NaN` meningkat saat window makin besar (`MA-20` paling rendah, `MA-240` paling tinggi).
- Terdapat bukti no-lookahead dari sample verifikasi.
- Date range final dan total trading days tercetak di log.
- NaN distribution per kolom MA tercetak di log.

## 7) Validation Checklist

- Print date range setelah backfill: `min_date -> max_date` dan jumlah trading days.
- Assert row count input == output.
- Assert kolom original bebas NaN.
- Tampilkan minimal 5 sample row untuk satu `(broker, ticker)` yang history-nya cukup.
- Simpan ke `master_broker_enriched.parquet`.

## 8) Level 1 Data Mart Handoff (Current MVP)

Section ini menjabarkan kontrak output datamart yang saat ini dipakai downstream setelah baseline enrichment.

### Data Mart Contract

| Item | Value |
|---|---|
| Data mart file | `data/Level_1_Features/broksum_datamart.parquet` |
| Grain | `1 row = 1 date + 1 broker + 1 ticker` |
| Composite key | `(date, broker, ticker)` |
| Default rolling windows | `5, 20, 60` (dapat diubah via argumen generator) |
| Current schema size (MVP) | `106 columns` |
| Upstream dependency | `broksum_bybroker.parquet`, `yfinance_daily.parquet`, `yfinance_1h.parquet`, `yfinance_4h.parquet`, `master_emiten.parquet` |
| Downstream usage | Training table join + label generation (MVP objective open 09:00 -> close 10:00, TP 3%) |

### Data Mart Field Dictionary (Grouped)

| Group | Columns | Keterangan |
|---|---|---|
| Identity / Key | `broker`, `ticker`, `date`, `broker_type`, `scraped_at` | Identitas observasi dan metadata source. |
| Raw Broker Activity | `market_breadth`, `gross_buy`, `gross_sell`, `total_net_buy`, `gross_turnover`, `buy_volume`, `sell_volume`, `net_volume`, `buy_freq`, `sell_freq`, `avg_buy_price`, `avg_sell_price` | Kolom broker activity hasil normalisasi dari raw broksum. |
| Base Derived Ratios | `abs_net_buy`, `net_flow_ratio`, `buy_sell_val_ratio`, `buy_sell_vol_ratio`, `buy_sell_freq_ratio`, `total_trades`, `net_buy_per_trade`, `churn_ratio`, `avg_spread_price`, `avg_spread_pct` | Rasio mikrostruktur untuk intensitas beli/jual dan churn. |
| Context / Share | `broker_day_abs_flow`, `ticker_day_abs_flow`, `market_day_abs_flow`, `broker_ticker_specificity`, `broker_market_share`, `ticker_market_share` | Konteks kontribusi broker/ticker relatif terhadap pasar hari yang sama. |
| Temporal: Net Buy | `total_net_buy_lag1`, `total_net_buy_ma_{5,20,60}`, `total_net_buy_std_{5,20,60}`, `total_net_buy_z_{5,20,60}`, `total_net_buy_velocity_{5,20,60}` | Fitur temporal no-lookahead (`shift(1)` + rolling). |
| Temporal: Flow/Churn | `net_flow_ratio_lag1`, `net_flow_ratio_ma_{5,20,60}`, `churn_ratio_lag1`, `churn_ratio_ma_{5,20,60}` | Stabilitas flow dan churn terhadap baseline historis. |
| Market Daily (Raw/Return) | `yf_daily_open`, `yf_daily_high`, `yf_daily_low`, `yf_daily_close`, `yf_daily_volume`, `yf_daily_prev_close`, `yf_daily_ret_oc`, `yf_daily_ret_cc`, `yf_daily_gap_open_prev_close`, `yf_daily_range_pct`, `yf_daily_turnover` | Ringkasan harian OHLCV per ticker untuk context market. |
| Market Daily (Rolling) | `yf_daily_close_ma_{5,20,60}`, `yf_daily_close_z_{5,20,60}`, `yf_daily_volume_ma_{5,20,60}`, `yf_daily_volume_velocity_{5,20,60}` | Baseline dan deviasi harga/volume harian per ticker. |
| Market Intraday 1H | `yf_1h_open`, `yf_1h_high`, `yf_1h_low`, `yf_1h_close`, `yf_1h_volume`, `yf_1h_bar_count`, `yf_1h_ret_oc`, `yf_1h_range_pct`, `yf_1h_first_bar_ret`, `yf_1h_two_bar_ret`, `yf_1h_turnover` | Agregasi intraday 1H menjadi metrik per hari per ticker. |
| Market Intraday 4H | `yf_4h_open`, `yf_4h_high`, `yf_4h_low`, `yf_4h_close`, `yf_4h_volume`, `yf_4h_bar_count`, `yf_4h_ret_oc`, `yf_4h_range_pct`, `yf_4h_first_bar_ret`, `yf_4h_two_bar_ret`, `yf_4h_turnover` | Agregasi intraday 4H menjadi metrik per hari per ticker. |
| Cross-Source Interaction | `net_buy_to_yf_daily_turnover`, `abs_net_buy_to_yf_daily_turnover`, `net_buy_to_yf_1h_turnover`, `net_buy_to_yf_4h_turnover` | Mengukur skala net buy broker terhadap turnover market. |
| Coverage Flags | `has_yf_daily`, `has_yf_1h`, `has_yf_4h` | Penanda ketersediaan data market per source (0/1). |

### Handoff Acceptance (Layer 2 -> Layer 3)

| Check | Kriteria |
|---|---|
| Uniqueness | Tidak ada duplikat pada `(date, broker, ticker)` |
| Required identity | `date`, `broker`, `ticker`, `total_net_buy` harus terisi |
| Temporal integrity | Fitur rolling memakai basis `H-1` (no look-ahead) |
| Market coverage | Kolom `has_yf_*` tersedia dan bukan semua nol |
| Sort consistency | Output final tersortir by `date, broker, ticker` |

## 9) Full Level 1 Column Dictionary (106 Columns)

Table ini dijabarkan dari schema aktual `data/Level_1_Features/broksum_datamart.parquet` pada saat dokumen ini diperbarui.

| Column | Type | Group | Definition |
|---|---|---|---|
| `broker` | `large_string` | Identity / Key | Broker code (uppercase). |
| `ticker` | `large_string` | Identity / Key | Ticker code normalized without .JK suffix. |
| `date` | `timestamp[us]` | Identity / Key | Trading date (daily grain). |
| `broker_type` | `large_string` | Identity / Key | Broker classification label when available. |
| `market_breadth` | `int64` | Raw Broker Activity | Market breadth value from raw broksum snapshot. |
| `gross_buy` | `double` | Raw Broker Activity | Total buy value by broker-ticker-date. |
| `gross_sell` | `double` | Raw Broker Activity | Total sell value by broker-ticker-date. |
| `total_net_buy` | `double` | Raw Broker Activity | Net buy value (gross_buy - gross_sell). |
| `gross_turnover` | `double` | Raw Broker Activity | Total turnover (gross_buy + gross_sell). |
| `buy_volume` | `double` | Raw Broker Activity | Total buy volume. |
| `sell_volume` | `double` | Raw Broker Activity | Total sell volume. |
| `net_volume` | `double` | Raw Broker Activity | Net volume (buy_volume - sell_volume). |
| `buy_freq` | `int64` | Raw Broker Activity | Buy transaction frequency/count. |
| `sell_freq` | `int64` | Raw Broker Activity | Sell transaction frequency/count. |
| `avg_buy_price` | `double` | Raw Broker Activity | Average buy price. |
| `avg_sell_price` | `double` | Raw Broker Activity | Average sell price. |
| `scraped_at` | `timestamp[ns]` | Identity / Key | Raw source ingestion timestamp. |
| `abs_net_buy` | `double` | Base Derived Ratios | Absolute value of total_net_buy. |
| `net_flow_ratio` | `double` | Base Derived Ratios | total_net_buy / gross_turnover. |
| `buy_sell_val_ratio` | `double` | Base Derived Ratios | gross_buy / gross_sell. |
| `buy_sell_vol_ratio` | `double` | Base Derived Ratios | buy_volume / sell_volume. |
| `buy_sell_freq_ratio` | `double` | Base Derived Ratios | buy_freq / sell_freq. |
| `total_trades` | `int64` | Base Derived Ratios | buy_freq + sell_freq. |
| `net_buy_per_trade` | `double` | Base Derived Ratios | total_net_buy / total_trades. |
| `churn_ratio` | `double` | Base Derived Ratios | gross_turnover / abs_net_buy. |
| `avg_spread_price` | `double` | Base Derived Ratios | avg_sell_price - avg_buy_price. |
| `avg_spread_pct` | `double` | Base Derived Ratios | avg_spread_price / avg_buy_price. |
| `broker_day_abs_flow` | `double` | Context / Share | Total abs_net_buy per (date, broker). |
| `ticker_day_abs_flow` | `double` | Context / Share | Total abs_net_buy per (date, ticker). |
| `market_day_abs_flow` | `double` | Context / Share | Total abs_net_buy per date across market. |
| `broker_ticker_specificity` | `double` | Context / Share | abs_net_buy / broker_day_abs_flow. |
| `broker_market_share` | `double` | Context / Share | broker_day_abs_flow / market_day_abs_flow. |
| `ticker_market_share` | `double` | Context / Share | ticker_day_abs_flow / market_day_abs_flow. |
| `total_net_buy_lag1` | `double` | Temporal: Net Buy | Lag-1 value of total_net_buy per (broker,ticker). |
| `total_net_buy_ma_5` | `double` | Temporal: Net Buy | Rolling mean of `total_net_buy_lag1` over 5 rows per (broker,ticker). |
| `total_net_buy_std_5` | `double` | Temporal: Net Buy | Rolling std dev of `total_net_buy_lag1` over 5 rows per (broker,ticker). |
| `total_net_buy_z_5` | `double` | Temporal: Net Buy | Z-score of current `total_net_buy` vs MA/STD window 5. |
| `total_net_buy_velocity_5` | `double` | Temporal: Net Buy | Ratio `total_net_buy / total_net_buy_ma_5`. |
| `total_net_buy_ma_20` | `double` | Temporal: Net Buy | Rolling mean of `total_net_buy_lag1` over 20 rows per (broker,ticker). |
| `total_net_buy_std_20` | `double` | Temporal: Net Buy | Rolling std dev of `total_net_buy_lag1` over 20 rows per (broker,ticker). |
| `total_net_buy_z_20` | `double` | Temporal: Net Buy | Z-score of current `total_net_buy` vs MA/STD window 20. |
| `total_net_buy_velocity_20` | `double` | Temporal: Net Buy | Ratio `total_net_buy / total_net_buy_ma_20`. |
| `total_net_buy_ma_60` | `double` | Temporal: Net Buy | Rolling mean of `total_net_buy_lag1` over 60 rows per (broker,ticker). |
| `total_net_buy_std_60` | `double` | Temporal: Net Buy | Rolling std dev of `total_net_buy_lag1` over 60 rows per (broker,ticker). |
| `total_net_buy_z_60` | `double` | Temporal: Net Buy | Z-score of current `total_net_buy` vs MA/STD window 60. |
| `total_net_buy_velocity_60` | `double` | Temporal: Net Buy | Ratio `total_net_buy / total_net_buy_ma_60`. |
| `net_flow_ratio_lag1` | `double` | Temporal: Flow/Churn | Lag-1 value of net_flow_ratio per (broker,ticker). |
| `net_flow_ratio_ma_5` | `double` | Temporal: Flow/Churn | Rolling mean of `net_flow_ratio_lag1` over 5 rows per (broker,ticker). |
| `net_flow_ratio_ma_20` | `double` | Temporal: Flow/Churn | Rolling mean of `net_flow_ratio_lag1` over 20 rows per (broker,ticker). |
| `net_flow_ratio_ma_60` | `double` | Temporal: Flow/Churn | Rolling mean of `net_flow_ratio_lag1` over 60 rows per (broker,ticker). |
| `churn_ratio_lag1` | `double` | Temporal: Flow/Churn | Lag-1 value of churn_ratio per (broker,ticker). |
| `churn_ratio_ma_5` | `double` | Temporal: Flow/Churn | Rolling mean of `churn_ratio_lag1` over 5 rows per (broker,ticker). |
| `churn_ratio_ma_20` | `double` | Temporal: Flow/Churn | Rolling mean of `churn_ratio_lag1` over 20 rows per (broker,ticker). |
| `churn_ratio_ma_60` | `double` | Temporal: Flow/Churn | Rolling mean of `churn_ratio_lag1` over 60 rows per (broker,ticker). |
| `yf_daily_open` | `double` | Market Daily (Raw/Return) | Daily open price. |
| `yf_daily_high` | `double` | Market Daily (Raw/Return) | Daily high price. |
| `yf_daily_low` | `double` | Market Daily (Raw/Return) | Daily low price. |
| `yf_daily_close` | `double` | Market Daily (Raw/Return) | Daily close price. |
| `yf_daily_volume` | `double` | Market Daily (Raw/Return) | Daily traded volume. |
| `yf_daily_prev_close` | `double` | Market Daily (Raw/Return) | Previous day close (lag1). |
| `yf_daily_ret_oc` | `double` | Market Daily (Raw/Return) | (close - open) / open. |
| `yf_daily_ret_cc` | `double` | Market Daily (Raw/Return) | (close - prev_close) / prev_close. |
| `yf_daily_gap_open_prev_close` | `double` | Market Daily (Raw/Return) | (open - prev_close) / prev_close. |
| `yf_daily_range_pct` | `double` | Market Daily (Raw/Return) | (high - low) / open. |
| `yf_daily_turnover` | `double` | Market Daily (Raw/Return) | close * volume. |
| `yf_daily_close_ma_5` | `double` | Market Daily (Rolling) | Rolling mean of prior daily close over 5 rows per ticker. |
| `yf_daily_close_z_5` | `double` | Market Daily (Rolling) | Z-score of daily close against rolling close baseline window 5. |
| `yf_daily_close_ma_20` | `double` | Market Daily (Rolling) | Rolling mean of prior daily close over 20 rows per ticker. |
| `yf_daily_close_z_20` | `double` | Market Daily (Rolling) | Z-score of daily close against rolling close baseline window 20. |
| `yf_daily_close_ma_60` | `double` | Market Daily (Rolling) | Rolling mean of prior daily close over 60 rows per ticker. |
| `yf_daily_close_z_60` | `double` | Market Daily (Rolling) | Z-score of daily close against rolling close baseline window 60. |
| `yf_daily_volume_ma_5` | `double` | Market Daily (Rolling) | Rolling mean of prior daily volume over 5 rows per ticker. |
| `yf_daily_volume_velocity_5` | `double` | Market Daily (Rolling) | Ratio `yf_daily_volume / yf_daily_volume_ma_5`. |
| `yf_daily_volume_ma_20` | `double` | Market Daily (Rolling) | Rolling mean of prior daily volume over 20 rows per ticker. |
| `yf_daily_volume_velocity_20` | `double` | Market Daily (Rolling) | Ratio `yf_daily_volume / yf_daily_volume_ma_20`. |
| `yf_daily_volume_ma_60` | `double` | Market Daily (Rolling) | Rolling mean of prior daily volume over 60 rows per ticker. |
| `yf_daily_volume_velocity_60` | `double` | Market Daily (Rolling) | Ratio `yf_daily_volume / yf_daily_volume_ma_60`. |
| `yf_1h_open` | `double` | Market Intraday 1H | First 1H open aggregated to day. |
| `yf_1h_high` | `double` | Market Intraday 1H | Max 1H high in day. |
| `yf_1h_low` | `double` | Market Intraday 1H | Min 1H low in day. |
| `yf_1h_close` | `double` | Market Intraday 1H | Last 1H close in day. |
| `yf_1h_volume` | `double` | Market Intraday 1H | Sum 1H volume in day. |
| `yf_1h_bar_count` | `double` | Market Intraday 1H | Number of 1H bars in day. |
| `yf_1h_ret_oc` | `double` | Market Intraday 1H | (close - open) / open at 1H-day aggregate. |
| `yf_1h_range_pct` | `double` | Market Intraday 1H | (high - low) / open at 1H-day aggregate. |
| `yf_1h_first_bar_ret` | `double` | Market Intraday 1H | Return of first intraday bar vs day open. |
| `yf_1h_two_bar_ret` | `double` | Market Intraday 1H | Return of second intraday close vs day open. |
| `yf_1h_turnover` | `double` | Market Intraday 1H | close * volume at 1H-day aggregate. |
| `yf_4h_open` | `double` | Market Intraday 4H | First 4H open aggregated to day. |
| `yf_4h_high` | `double` | Market Intraday 4H | Max 4H high in day. |
| `yf_4h_low` | `double` | Market Intraday 4H | Min 4H low in day. |
| `yf_4h_close` | `double` | Market Intraday 4H | Last 4H close in day. |
| `yf_4h_volume` | `double` | Market Intraday 4H | Sum 4H volume in day. |
| `yf_4h_bar_count` | `double` | Market Intraday 4H | Number of 4H bars in day. |
| `yf_4h_ret_oc` | `double` | Market Intraday 4H | (close - open) / open at 4H-day aggregate. |
| `yf_4h_range_pct` | `double` | Market Intraday 4H | (high - low) / open at 4H-day aggregate. |
| `yf_4h_first_bar_ret` | `double` | Market Intraday 4H | Return of first intraday bar vs day open. |
| `yf_4h_two_bar_ret` | `double` | Market Intraday 4H | Return of second intraday close vs day open. |
| `yf_4h_turnover` | `double` | Market Intraday 4H | close * volume at 4H-day aggregate. |
| `net_buy_to_yf_daily_turnover` | `double` | Cross-Source Interaction | total_net_buy / yf_daily_turnover. |
| `abs_net_buy_to_yf_daily_turnover` | `double` | Cross-Source Interaction | abs_net_buy / yf_daily_turnover. |
| `net_buy_to_yf_1h_turnover` | `double` | Cross-Source Interaction | total_net_buy / yf_1h_turnover. |
| `net_buy_to_yf_4h_turnover` | `double` | Cross-Source Interaction | total_net_buy / yf_4h_turnover. |
| `has_yf_daily` | `int8` | Coverage Flags | 1 if daily market feature exists for row, else 0. |
| `has_yf_1h` | `int8` | Coverage Flags | 1 if 1H market feature exists for row, else 0. |
| `has_yf_4h` | `int8` | Coverage Flags | 1 if 4H market feature exists for row, else 0. |
