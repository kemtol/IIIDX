# Model Handoff

## Current BSJP Research State (2026-05-09)

- Current clean research candidate: `model/BSJP/v23b_t1audit2_clean/`.
- Objective: BSJP `close10` — entry close 15:xx T, exit open 10:xx T+1.
- Status: research-only, not production/inference-ready.
- Locked OOT artifact: 2025-11-17 -> 2026-04-23.
- Locked OOT metrics: AUC 0.5346, cum net +207.9%, MaxDD -24.2%, best iteration 5, overfit gap 0.0616.
- Current quick-win policy candidate: k=2 / max weight 25% / cost cap 3% / q=.85, locked OOT +255.0%, MaxDD -17.9%.
- Conservative policy candidate: k=2 / max weight 20% / q=.90, locked OOT +184.7%, MaxDD -13.5%.
- No-lookahead audit: `_LOG/v23b_t1audit2_clean_no_lookahead_audit_20260509.json`; hard failures all false.
- Latest provisional calendar extension ends at 2026-05-06. 2026-05-09 is Saturday and 2026-05-08 entry is not closed yet.
- Remaining blocker: robustness/overfit validation via full rolling-retrain, not a confirmed leakage failure in the current audit scope.

## Read Order

1. `program.md`
2. `model/BSJP/LATEST.md`
3. Latest `_MEMORY/YYYYMMDDHHMMSS.md`
4. `edges/bsjp_overnight_sl2/edge.md`

Do not use v19d/v20 headline returns as credibility benchmarks. They are now considered inflated by confirmed same-day broker leakage and ARA fillability assumptions.
