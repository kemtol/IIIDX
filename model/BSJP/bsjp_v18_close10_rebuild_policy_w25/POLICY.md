# BSJP v18 Close10 Rebuild — Paper-Trade Policy

## Policy

Use the same model as `model/BSJP/bsjp_v18_close10_rebuild/`, with a lower concentration cap:

| Field | Value |
|---|---:|
| Objective | close10: entry close@15 T, exit open@10 T+1 |
| Policy mode | threshold |
| `p_cut` | 0.035 |
| Adaptive threshold | enabled |
| Adaptive threshold quantile | 0.85 |
| Conviction top-k | 3 |
| Max positions | 3 |
| Max weight per name | 0.25 |
| Execution model | market |
| TP/SL labels | `tp_pct=0.035`, `sl_pct=-0.015` |

## Why This Policy

Baseline `bsjp_v18_close10_rebuild` used `max_weight=0.34`, producing higher return but deeper OOT and Monte Carlo drawdowns.

The `threshold_k3_w25` policy keeps the same model and top-3 structure but caps each name at 25%, reducing concentration risk.

## OOT Result

| Metric | Baseline k3/w34 | Policy k3/w25 |
|---|---:|---:|
| Mean daily net | 1.30% | 0.98% |
| Cumulative net | 2.126x | 1.431x |
| MaxDD | -35.6% | -28.2% |
| Worst day | -10.3% | -7.6% |
| Days <= -5% | 5 | 2 |
| Days <= -8% | 1 | 0 |

## Monte Carlo

10,000 paths, block bootstrap, block size 5.

| Horizon | Median Terminal | P(loss) | Mean MaxDD | P(MaxDD <= -30%) |
|---|---:|---:|---:|---:|
| 100d | 2.57x | 1.78% | -19.6% | 7.61% |
| 252d | 11.34x | 0.04% | -25.2% | 22.0% |

## Artifacts

- `portfolio_daily.parquet`
- `policy.json`
- `policy_sensitivity.csv` in the source model directory
- `pnl_20d.png`, `pnl_50d.png`, `pnl_100d.png`
- `monte_carlo/monte_summary_metrics.csv`
- `monte_carlo/monte_equity_fan_*.png`
- `monte_carlo/monte_maxdd_hist_*.png`
- `monte_carlo/monte_return_cdf_*.png`
