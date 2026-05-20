# BSJP v25 Clean T-1 Quick Candidate

Folder ini adalah **current BSJP research candidate** per 2026-05-10.

Nama folder:

`model/BSJP/v25_clean_t1quick_nl31_md100_l2.0_market_k3_w25/`

Status:

- **PASS as research candidate**
- **Not production-ready yet**
- Continuation harus mulai dari folder ini, bukan dari v23/v24/v25 experiment lain.
- Referensi production/stable lama masih disimpan di `model/BSJP/v24b_final_production/`.
- Experiment lain sudah diarsipkan ke `model/BSJP/_ARCH/20260510_model_cleanup/`.

## 1. Objective

Strategi:

- Product: BSJP
- Objective: `close10`
- Entry: close 15:xx pada hari T
- Exit: open 10:xx pada hari T+1
- Model: binary LightGBM classifier
- Target label: `label_tp`, yaitu return setelah cost melewati threshold TP.

Northstar:

- Positive net expectancy after costs.
- Model hanya boleh memakai feature yang tersedia sebelum keputusan entry sore.
- No lookahead bias.
- OOT memakai 100 trading days terakhir secara chronological split.

Operational context:

- Order window user: sekitar 15:30 sampai 15:45 WIB.
- Feature yang dipakai harus sudah tersedia sebelum window tersebut.
- Feature same-day/full-day yang baru diketahui setelah market close tidak boleh dipakai kecuali sudah digeser T-1 atau dibuktikan available sebelum decision time.

## 2. Data Input

Training input:

`data/Level_2_Datamart/training_datamart_bsjp_v25_clean_t1quick.parquet`

Audit file:

`data/Level_2_Datamart/training_datamart_bsjp_v25_clean_t1quick.parquet.audit.json`

Data summary:

| Metric | Value |
|---|---:|
| Rows | 427,828 |
| Trading days | 736 |
| Date min | 2023-03-06 |
| Date max | 2026-05-06 |
| OOT days | 100 |
| OOT range | 2025-11-26 to 2026-05-06 |
| Tickers | 769 |

Clean datamart status:

- Duplicate `(date, ticker)`: 0
- `inf`: 0
- Hard audit failures: none
- Low-coverage dates below 300 tickers were dropped.
- Label formula checks passed in the prior v25 audit:
  - `(exit_price - entry_price) / entry_price`
  - `label_tp`
  - `label_sl2`
  - reconstructed yfinance label parity

## 3. Retrospective: Why This Model Exists

The original v25 locked datamart had strong-looking candidates, but audit found high-risk feature families that could contain same-day/full-day leakage for a 15:30 decision process.

Unsafe families removed in the P0 clean build:

- `f2_*`
- `forensic_*`
- `v25_sector_*`
- `sector_ticker_count`

Specific dropped unsafe columns:

- `f2_inventory_decay`
- `f2_absorption_ratio`
- `f2_vol_surge_14h`
- `forensic_hhi_index`
- `forensic_max_aggression`
- `forensic_mean_aggression`
- `forensic_top_broker_share`
- `forensic_inventory_10d`
- `v25_sector_flow_share`
- `v25_sector_turnover_share`
- `sector_ticker_count`

The first clean retrain without these families was:

`model/BSJP/v25_clean_nl31_md100_l2.0_market_k3_w25/`

That baseline failed portfolio expectancy:

| Metric | Value |
|---|---:|
| OOT AUC | 0.5468 |
| OOT cumulative net | -57.35% |
| OOT MaxDD | -58.06% |
| Mean daily net | -0.81% |

Interpretation:

- The P0-clean model still had a ranking signal by AUC.
- But portfolio expectancy collapsed after the unsafe feature families were removed.
- Therefore, the next quick win was not to re-add unsafe same-day features, but to recover a small subset as shifted T-1 features.

## 4. T-1 Quick Feature Repair

This candidate adds four shifted-safe T-1 columns derived from the previously dropped families:

- `f2_inventory_decay_t1`
- `forensic_inventory_10d_t1`
- `v25_sector_turnover_share_t1`
- `v25_sector_flow_share_t1`

Safety rule:

- Date T uses previous available source row per ticker.
- Unshifted same-day v25 columns remain removed.

Coverage caveat:

- T-1 quick features have around 56.65% null coverage because source modules begin around 2024-10.
- This is acceptable for a quick candidate but should be improved before promotion.

T-1 feature importance in this run:

| Feature | Gain | Splits |
|---|---:|---:|
| `v25_sector_turnover_share_t1` | 82.60 | 3 |
| `v25_sector_flow_share_t1` | 22.84 | 1 |
| `f2_inventory_decay_t1` | 11.49 | 2 |
| `forensic_inventory_10d_t1` | 0.00 | 0 |

Interpretation:

- The sector T-1 features helped restore portfolio expectancy.
- The model still mostly relies on pre14 intraday structure.
- `forensic_inventory_10d_t1` did not contribute in this run.

## 5. Training Config

Command used:

```bash
env PYTHONUNBUFFERED=1 python3 edges/bsjp_overnight_sl2/scripts/train_lightgbm.py \
  --training-path data/Level_2_Datamart/training_datamart_bsjp_v25_clean_t1quick.parquet \
  --output-dir model/BSJP/v25_clean_t1quick_nl31_md100_l2.0_market_k3_w25 \
  --feature-prune-top-n 0 \
  --execution-model market \
  --max-pre14-market-cost-est 0.030 \
  --min-entry-price-filter 500 \
  --policy-modes threshold \
  --p-cut-grid 0.035 \
  --max-positions-grid 3 \
  --max-weight-grid 0.25 \
  --oot-valid-days 100 \
  --num-leaves 31 \
  --max-depth 5 \
  --min-data-in-leaf 100 \
  --lambda-l1 0.5 \
  --lambda-l2 1.5 \
  --min-gain-to-split 0.05
```

LightGBM params:

| Param | Value |
|---|---:|
| `n_estimators` | 4000 |
| `learning_rate` | 0.02 |
| `num_leaves` | 31 |
| `max_depth` | 5 |
| `min_data_in_leaf` | 100 |
| `feature_fraction` | 0.6 |
| `bagging_fraction` | 0.8 |
| `bagging_freq` | 1 |
| `lambda_l1` | 0.5 |
| `lambda_l2` | 1.5 |
| `min_gain_to_split` | 0.05 |
| `objective` | binary |
| `early_stopping_rounds` | 200 |
| `best_iteration` | 4 |

Policy:

| Policy setting | Value |
|---|---:|
| Mode | threshold |
| `p_cut` | 0.035 |
| Adaptive threshold | true |
| Adaptive quantile | 0.85 |
| Max positions | 3 |
| Max weight per name | 25% |
| Min entry price | 500 |
| Max pre14 market cost estimate | 3% |
| Execution model | market |

Cost assumptions:

| Cost | Value |
|---|---:|
| Buy cost | 10 bps |
| Sell cost | 20 bps |
| Slippage per side | 5 bps |
| Roundtrip total | 0.40% |

## 6. Metrics

Status:

`PASS`

Core metrics:

| Metric | Value |
|---|---:|
| Train AUC | 0.5717 |
| OOT AUC | 0.5459 |
| Train AUCPR | 0.4016 |
| OOT AUCPR | 0.4017 |
| Overfit gap AUC | 0.0259 |
| OOT positive rate | 36.64% |
| Selected features | 71 |

Walk-forward AUC:

| Fold | Train days | Valid days | AUC | AUCPR |
|---|---:|---:|---:|---:|
| 1 | 556 | 20 | 0.5687 | 0.4299 |
| 2 | 576 | 20 | 0.5457 | 0.4261 |
| 3 | 596 | 20 | 0.5538 | 0.4244 |
| 4 | 616 | 20 | 0.5595 | 0.3986 |

OOT portfolio:

| Metric | Value |
|---|---:|
| Days | 100 |
| Mean daily net | +0.7803% |
| Mean daily gross | +2.1723% |
| Net expectancy per trade | +1.0404% |
| Win rate days | 50.0% |
| Cumulative net return | +95.59% |
| Max drawdown | -35.95% |
| Daily volatility | 4.68% |
| Mean effective threshold | 0.3641 |

Daily top-k quality:

| Top K | Precision mean | Hit rate any TP | Rows |
|---|---:|---:|---:|
| 5 | 53.0% | 88.0% | 500 |
| 10 | 48.0% | 98.0% | 1,000 |
| 20 | 46.55% | 99.0% | 2,000 |

## 7. PnL Snapshot

From `portfolio_daily.parquet`, compounded net through latest OOT date `2026-05-06`:

| Window | Return | Normalized PnL on Rp10m |
|---|---:|---:|
| 7 trading days | -1.88% | -Rp187k |
| 30 trading days | +22.31% | +Rp2.23m |
| 90 trading days | +49.78% | +Rp4.98m |

Last 7 OOT daily returns:

| Date | Net return |
|---|---:|
| 2026-04-27 | -1.50% |
| 2026-04-28 | +3.44% |
| 2026-04-29 | -2.65% |
| 2026-04-30 | -1.57% |
| 2026-05-04 | +1.20% |
| 2026-05-05 | -0.73% |
| 2026-05-06 | +0.05% |

Daily return distribution over OOT:

| Statistic | Net return |
|---|---:|
| Mean | +0.78% |
| Median | +0.01% |
| Min | -10.53% |
| 5th pct | -5.83% |
| 95th pct | +8.76% |
| Max | +13.61% |

Worst OOT days:

| Date | Net return |
|---|---:|
| 2026-03-06 | -10.53% |
| 2026-04-23 | -7.27% |
| 2026-04-24 | -6.70% |
| 2026-01-28 | -6.55% |
| 2026-01-30 | -6.01% |

Best OOT days:

| Date | Net return |
|---|---:|
| 2026-01-14 | +13.61% |
| 2025-12-03 | +11.44% |
| 2026-01-21 | +11.11% |
| 2026-01-07 | +10.65% |
| 2026-01-22 | +9.75% |

## 8. Monte Carlo

Monte Carlo was run on 2026-05-18 using block bootstrap over `portfolio_daily.parquet`:

```bash
python3 edges/bsjp_overnight_sl2/scripts/run_monte_carlo.py \
  model/BSJP/v25_clean_t1quick_nl31_md100_l2.0_market_k3_w25 \
  --n_paths 10000 \
  --block_size 5 \
  --horizons 100 252
```

Source:

- Rows: 100 OOT daily returns
- Date range: 2025-11-26 to 2026-05-06
- Daily return mean: +0.7803%
- Daily return std: 4.6766%
- Min / max daily return: -10.53% / +13.61%

Summary:

| Horizon | Median terminal | P5 terminal | P95 terminal | P(loss) | Mean MaxDD | Median MaxDD | P(MaxDD <= -30%) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100d | 2.03x | 0.73x | 5.73x | 12.88% | -35.09% | -33.32% | 61.79% |
| 252d | 5.78x | 1.13x | 29.64x | 3.98% | -46.43% | -45.18% | 93.45% |

Read:

- Terminal return distribution is positive, but drawdown risk is high.
- 100d median terminal is about 2.03x, but 100d P(loss) is 12.88%.
- 100d P(MaxDD <= -30%) is 61.79%, confirming the current policy is not yet operationally conservative.
- 252d projection compounds strongly but has very high drawdown exposure; use it as a stress diagnostic, not a promotion claim.

## 9. Feature Read

Top gain features:

| Rank | Feature | Gain | Splits |
|---:|---|---:|---:|
| 1 | `pre14_tick_pct` | 14488.05 | 30 |
| 2 | `pre14_range_pct` | 5149.29 | 16 |
| 3 | `pre14_vwap` | 3573.28 | 4 |
| 4 | `pre14_spread_cost_est` | 1608.95 | 5 |
| 5 | `pre14_pm_turnover` | 1460.74 | 11 |
| 6 | `pre14_close_vs_orb_low` | 969.82 | 5 |
| 7 | `pre14_pm_high_close_spread` | 748.83 | 2 |
| 8 | `pre14_close_orb_high_hold` | 709.74 | 1 |
| 9 | `pre14_orb_upper_wick_pct` | 623.13 | 2 |
| 10 | `pre14_volume_14h` | 224.64 | 1 |

Plain read:

- Main signal is still pre14 intraday behavior.
- T-1 sector flow/turnover contributes but is not dominant.
- No single T-1 quick feature dominates the model.
- `best_iteration=4` is low. This is not automatically leakage, but it does mean the model is shallow/early-stopped and should be tested for stability.

## 10. Artifact List

Files in this folder:

| File | Purpose |
|---|---|
| `README.md` | This handoff document |
| `metrics.json` | Primary machine-readable model metrics and config |
| `model_lightgbm_opening_tp3.txt` | Trained LightGBM model artifact |
| `feature_importance.csv` | Feature importance by gain and split |
| `valid_predictions.parquet` | OOT predictions with ranks and selected policy support columns |
| `portfolio_daily.parquet` | OOT daily portfolio simulation |
| `walkforward_metrics.csv` | Fold-level validation metrics |
| `pnl_20d.png` | PnL chart, last 20 OOT days |
| `pnl_50d.png` | PnL chart, last 50 OOT days |
| `pnl_100d.png` | PnL chart, full 100-day OOT |
| `monte_carlo/monte_config.json` | Monte Carlo configuration |
| `monte_carlo/monte_summary_metrics.csv` | Monte Carlo summary metrics |
| `monte_carlo/monte_equity_fan_100d.png` | 100d Monte Carlo equity fan |
| `monte_carlo/monte_equity_fan_252d.png` | 252d Monte Carlo equity fan |
| `monte_carlo/monte_maxdd_hist_100d.png` | 100d MaxDD histogram |
| `monte_carlo/monte_maxdd_hist_252d.png` | 252d MaxDD histogram |
| `monte_carlo/monte_return_cdf_100d.png` | 100d terminal return CDF |
| `monte_carlo/monte_return_cdf_252d.png` | 252d terminal return CDF |

Important external artifacts:

| Path | Purpose |
|---|---|
| `data/Level_2_Datamart/training_datamart_bsjp_v25_clean_t1quick.parquet` | Training datamart used by this model |
| `data/Level_2_Datamart/training_datamart_bsjp_v25_clean_t1quick.parquet.audit.json` | Clean datamart audit |
| `pipeline/audit/build_clean_v25_datamart.py` | Script that generated clean and T-1 quick datamarts |
| `_MEMORY/20260510192015.md` | Latest session handoff |
| `model/BSJP/LATEST.md` | Long-form model chronology |
| `model/BSJP/_ARCH/20260510_model_cleanup/README.md` | Archive cleanup manifest |

## 11. Known Caveats

Do not promote this model yet without addressing these:

- MaxDD is still large at -35.95%.
- Monte Carlo confirms high drawdown risk: 100d P(MaxDD <= -30%) is 61.79%.
- OOT has already been repeatedly observed during research, so avoid further direct tuning against this same 100-day OOT.
- T-1 quick feature coverage is incomplete because upstream source modules begin around 2024-10.
- `best_iteration=4` means the final model is very shallow; validate across parameter perturbations and rolling retrain.
- The model is not yet calibrated into Go inference or production DuckDB feature generation.
- Feature availability for all production-time columns must be verified against the 15:30 to 15:45 operational window.
- IPOT websocket fallback exists but current `ws_trend_1m.duckdb` source coverage was only 159 tickers during simulation.

## 12. What Next

P0:

1. Run rolling-retrain validation for this exact clean T-1 quick candidate.
2. Verify T-1 shift safety around holidays and missing source dates.
3. Test conservative policies without changing features:
   - `max_positions=2`
   - `max_weight=0.20`
   - stricter adaptive threshold or quantile if supported
4. Decide whether MaxDD -35.95% and 100d MC P(MaxDD <= -30%) of 61.79% are acceptable. If not, policy tightening is mandatory before deployment discussion.

P1:

1. Rebuild more previously unsafe features as availability-safe T-1 variants.
2. Improve source history/coverage for T-1 feature families.
3. Add tests for:
   - duplicate key removal
   - low coverage date dropping
   - T-1 shift behavior
   - no unshifted unsafe v25 columns in clean datamart
4. Compare current `k=3/w25` policy against `k=2/w25` and `k=2/w20`.

P2:

1. Formalize `build_clean_v25_datamart.py` into the reproducible training pipeline.
2. Produce a promotion checklist if rolling-retrain passes.
3. Only after research promotion, map selected features into inference and validate training-inference parity.

## 13. Continuation Instructions

For the next agent:

1. Read `program.md`.
2. Read `model/BSJP/LATEST.md`.
3. Read `_MEMORY/20260510192015.md`.
4. Read this README.
5. Continue from:

```text
model/BSJP/v25_clean_t1quick_nl31_md100_l2.0_market_k3_w25/
```

Do not continue from archived v23/v25 folders unless the user explicitly asks for forensic comparison.

If doing new model work, write results to a new folder under `model/BSJP/` and update:

- `model/BSJP/LATEST.md`
- latest `_MEMORY/YYYYMMDDHHMMSS.md`
- this README if the current candidate remains relevant
