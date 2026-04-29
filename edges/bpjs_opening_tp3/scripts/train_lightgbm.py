#!/usr/bin/env python3
"""
Train LightGBM for BPJS opening->10:00 objective with strict safeguards:
- Hard minimum history gate (>=120 trading days)
- No lookahead leakage in feature set
- Chronological split only (walk-forward + strict OOT window)
- Portfolio-level policy search (probability + risk sizing)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


# edges/bpjs_opening_tp3/scripts/<file>.py → parents[3] = idx/
IDX_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = IDX_DIR / "data"
DATAMART_DIR = DATA_DIR / "Level_2_Datamart"

DEFAULT_TRAINING_PATH = DATAMART_DIR / "training_datamart_bpjs_intraday.parquet"
DEFAULT_OUTPUT_DIR = IDX_DIR / "model" / "BPJS" / "bpjs_lgbm_opening_tp3"
DEFAULT_FEATURE_IMPORTANCE_PATH = IDX_DIR / "model" / "BPJS" / "bpjs_lgbm_opening_tp3_v9" / "feature_importance.csv"

TARGET_COL = "label_tp"
ID_COLS = {"date", "ticker"}
OUTCOME_COLS = {
    "entry_price_opening",
    "high_to_cutoff",
    "low_to_cutoff",
    "close_to_cutoff",
    "volume",
    "trade_date",
    "entry_datetime",
    "entry_hm",
    "bars_until_cutoff",
    "max_return_to_cutoff",
    "min_return_to_cutoff",
    "close_return_to_cutoff",
    "label_tp",
    "label_sl3",
    "label_name",
}
EXECUTION_PRICE_COLS = [
    "entry_price_opening",
    "high_to_cutoff",
    "low_to_cutoff",
    "close_to_cutoff",
]

# Hard fallback for broker-confluence features when dynamic ranking file is unavailable.
# Column names reflect L1 prefix schema (flow_, tfl_, ctx_) → L2 aggregate suffix (_sum, _mean).
HARD_PRUNED_BROKER_FEATURES = [
    "ctx_broker_market_share_mean",
    "mg_flow_churn_ratio",
    "localfund_buy_freq_ratio_to_ma30",
    "localfund_buy_freq_std30",
    "localfund_netbuy_std20",
    "flow_churn_ratio_mean",
    "ctx_broker_ticker_specificity_mean",
    "tfl_churn_ratio_ma_5_mean",
    "tfl_churn_ratio_ma_20_sum",
    "localfund_buy_freq_z30",
    "tfl_churn_ratio_ma_60_mean",
    "bandar_buy_freq_mean",
    "localfund_netbuy_ma20",
    "tfl_net_flow_ratio_ma_5_sum",
    "flow_churn_ratio_sum",
    "flow_sell_freq_mean",
    "tfl_churn_ratio_ma_5_sum",
    "localfund_buy_freq_ma30",
    "localfund_netbuy_z20",
    "localfund_netbuy_3d_sum",
    "flow_net_flow_ratio_mean",
    "tfl_churn_ratio_ma_60_sum",
    "flow_abs_net_buy_mean",
    "localfund_netbuy_7d_sum",
    "flow_net_flow_ratio_sum",
    "flow_buy_freq_mean",
    "tfl_net_buy_z_5_sum",
    "tfl_net_buy_ma_5_mean",
    "tfl_net_flow_ratio_ma_5_mean",
    "tfl_net_buy_velocity_5_mean",
    "ctx_ticker_market_share_mean",
    "tfl_churn_ratio_ma_20_mean",
    "tfl_net_buy_z_20_mean",
    "tfl_net_flow_ratio_ma_60_mean",
    "tfl_net_buy_ma_20_sum",
]


@dataclass
class PolicyConfig:
    mode: str
    p_cut: float
    max_positions: int
    max_weight_per_name: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM for BPJS TP3 opening objective.")
    parser.add_argument("--training-path", type=Path, default=DEFAULT_TRAINING_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=42)

    # Hard gates
    parser.add_argument("--min-trading-days", type=int, default=120)
    parser.add_argument("--oot-valid-days", type=int, default=20)
    parser.add_argument("--walkforward-folds", type=int, default=4)
    parser.add_argument("--walkforward-valid-days", type=int, default=20)
    parser.add_argument("--earlystop-valid-days", type=int, default=20)
    parser.add_argument("--feature-prune-top-n", type=int, default=35)
    parser.add_argument("--feature-importance-path", type=Path, default=DEFAULT_FEATURE_IMPORTANCE_PATH)

    # Policy search grid
    parser.add_argument("--p-cut-grid", type=str, default="0.035")
    parser.add_argument("--policy-modes", type=str, default="threshold")
    parser.add_argument("--max-positions-grid", type=str, default="3")
    parser.add_argument("--max-weight-grid", type=str, default="0.34")
    parser.add_argument("--capital-idr", type=float, default=10_000_000.0)
    parser.add_argument("--min-allocation-idr", type=float, default=100_000.0)
    parser.add_argument("--min-policy-trading-days", type=int, default=1)
    parser.add_argument("--conviction-top-k", type=int, default=3)
    parser.add_argument("--conviction-min-proba", type=float, default=0.035)
    parser.add_argument(
        "--use-adaptive-threshold",
        action="store_true",
        default=True,
        help="Use per-day adaptive threshold max(conviction_min_proba, daily quantile).",
    )
    parser.add_argument(
        "--disable-adaptive-threshold",
        action="store_false",
        dest="use_adaptive_threshold",
        help="Disable adaptive threshold and use fixed absolute floor logic.",
    )
    parser.add_argument(
        "--adaptive-threshold-quantile",
        type=float,
        default=0.85,
        help="Daily prediction quantile used when adaptive threshold is enabled.",
    )

    # Cost assumptions
    parser.add_argument("--buy-cost-bps", type=float, default=10.0)
    parser.add_argument("--sell-cost-bps", type=float, default=20.0)
    parser.add_argument("--slippage-bps-per-side", type=float, default=5.0)
    parser.add_argument("--tp-pct", type=float, default=0.035, help="Take profit threshold, e.g., 0.035 for 3.5% (Aggressive Scalper)")
    parser.add_argument("--sl-pct", type=float, default=-0.015, help="Stop loss threshold, e.g., -0.015 for -1.5% (Aggressive Scalper)")

    # LightGBM guardrail params
    parser.add_argument("--n-estimators", type=int, default=4000)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-data-in-leaf", type=int, default=500)
    parser.add_argument("--feature-fraction", type=float, default=0.6)
    parser.add_argument("--bagging-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-freq", type=int, default=1)
    parser.add_argument("--lambda-l1", type=float, default=2.0)
    parser.add_argument("--lambda-l2", type=float, default=2.0)
    parser.add_argument("--min-gain-to-split", type=float, default=0.1)
    parser.add_argument(
        "--scale-pos-weight",
        type=float,
        default=-1.0,
        help="<=0 uses auto neg/pos ratio; >0 uses fixed value (set 1.0 to disable reweighting).",
    )
    parser.add_argument(
        "--pos-weight-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for auto scale_pos_weight (applied after auto calculation).",
    )
    parser.add_argument(
        "--oversample-ratio",
        type=float,
        default=0.0,
        help="If >0, oversample positive class to achieve this ratio (positive/(positive+negative)).",
    )
    parser.add_argument("--early-stopping-rounds", type=int, default=200)
    parser.add_argument("--topk-list", type=str, default="5,10,20")
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    if not out:
        raise ValueError(f"Unable to parse int list from: {text}")
    return sorted(set(out))


def parse_float_list(text: str) -> list[float]:
    out: list[float] = []
    for token in text.split(","):
        token = token.strip()
        if token:
            out.append(float(token))
    if not out:
        raise ValueError(f"Unable to parse float list from: {text}")
    return sorted(set(out))


def parse_mode_list(text: str) -> list[str]:
    out = []
    for token in text.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token not in {"threshold", "rank", "pure"}:
            raise ValueError(f"Unsupported policy mode: {token}")
        out.append(token)
    if not out:
        raise ValueError("No policy mode parsed. Example: --policy-modes threshold,rank")
    return list(dict.fromkeys(out))


def load_ranked_feature_candidates(path: Path, top_n: int) -> tuple[list[str], str]:
    if top_n <= 0:
        return [], "disabled"

    if path.exists():
        try:
            fi = pd.read_csv(path)
            if "feature" in fi.columns:
                ranked = [str(x) for x in fi["feature"].dropna().tolist()]
                if ranked:
                    return ranked[:top_n], f"csv:{path}"
        except Exception:
            pass

    return HARD_PRUNED_BROKER_FEATURES[:top_n], "hardcoded_fallback"


def binary_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = pd.Series(y_true).astype(int)
    y_pred = (y_prob >= threshold).astype(int)
    metrics: dict[str, float] = {}
    if y_true.nunique() > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["aucpr"] = float(average_precision_score(y_true, y_prob))
        metrics["logloss"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
        metrics["brier"] = float(brier_score_loss(y_true, y_prob))
    else:
        metrics["auc"] = float("nan")
        metrics["aucpr"] = float("nan")
        metrics["logloss"] = float("nan")
        metrics["brier"] = float("nan")
    metrics["precision_at_0.5"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall_at_0.5"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1_at_0.5"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["positive_rate"] = float(y_true.mean())
    return metrics


def daily_topk_metrics(df: pd.DataFrame, k_list: list[int]) -> dict[str, dict[str, float]]:
    scored = df[["date", "ticker", TARGET_COL, "pred_proba"]].copy()
    scored = scored.sort_values(["date", "pred_proba"], ascending=[True, False]).reset_index(drop=True)
    out: dict[str, dict[str, float]] = {}
    for k in k_list:
        topk = scored.groupby("date", sort=False).head(k).copy()
        per_day = topk.groupby("date", sort=False)[TARGET_COL].agg(["mean", "max"]).reset_index()
        out[f"k={k}"] = {
            "precision_mean": float(per_day["mean"].mean()) if not per_day.empty else float("nan"),
            "hit_rate_any_tp": float(per_day["max"].mean()) if not per_day.empty else float("nan"),
            "days": int(per_day["date"].nunique()) if not per_day.empty else 0,
            "rows": int(len(topk)),
        }
    return out


def choose_feature_columns(df: pd.DataFrame, ranked_candidates: list[str] | None = None) -> list[str]:
    blocked = OUTCOME_COLS | ID_COLS
    numeric_cols = [c for c in df.columns if c not in blocked and pd.api.types.is_numeric_dtype(df[c])]

    if ranked_candidates:
        numeric_set = set(numeric_cols)
        ranked_cols = [c for c in ranked_candidates if c in numeric_set]
        if ranked_cols:
            return ranked_cols

    return numeric_cols


def _clip_series(s: pd.Series, q_low: float = 0.01, q_high: float = 0.99) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 10:
        return s
    lo = s.quantile(q_low)
    hi = s.quantile(q_high)
    return s.clip(lower=lo, upper=hi)


def build_risk_proxy(df: pd.DataFrame) -> pd.Series:
    risk_sources = []
    candidates = [
        "flow_churn_ratio_mean",
        "tfl_churn_ratio_ma_20_mean",
        "seller_ratio",
        "bandar_concentration_hhi",
        "localfund_concentration_hhi",
    ]
    for c in candidates:
        if c in df.columns:
            s = _clip_series(df[c])
            risk_sources.append(s.rank(pct=True))

    if "tfl_net_buy_z_20_mean" in df.columns:
        s = _clip_series(df["tfl_net_buy_z_20_mean"].abs())
        risk_sources.append(s.rank(pct=True))
    if "localfund_buy_freq_z30" in df.columns:
        s = _clip_series(df["localfund_buy_freq_z30"].abs())
        risk_sources.append(s.rank(pct=True))

    if not risk_sources:
        return pd.Series(np.full(len(df), 0.5), index=df.index)

    risk = pd.concat(risk_sources, axis=1).mean(axis=1, skipna=True)
    risk = risk.fillna(risk.median() if risk.notna().any() else 0.5)
    return risk.clip(0.0, 1.0)


def capped_weights(scores: np.ndarray, max_weight: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0 or scores.sum() <= 0:
        return np.zeros_like(scores)
    w = scores / scores.sum()
    if max_weight <= 0:
        return np.zeros_like(scores)
    max_weight = min(max_weight, 1.0)

    # Iterative projection with cap.
    for _ in range(10):
        over = w > max_weight
        if not over.any():
            break
        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight
        under = ~over
        if not under.any():
            break
        under_sum = float(w[under].sum())
        if under_sum <= 0:
            break
        w[under] += w[under] / under_sum * excess
    return np.clip(w, 0.0, max_weight)


def simulate_portfolio(
    scored_df: pd.DataFrame,
    policy: PolicyConfig,
    capital_idr: float,
    min_alloc_idr: float,
    total_cost_frac: float,
    conviction_top_k: int,
    conviction_min_proba: float,
    use_adaptive_threshold: bool,
    adaptive_threshold_quantile: float,
    tp_pct: float,
    sl_pct: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    required = {"date", "ticker", "pred_proba", TARGET_COL, "risk_norm"} | set(EXECUTION_PRICE_COLS)
    missing = sorted(required - set(scored_df.columns))
    if missing:
        raise ValueError(f"simulate_portfolio missing columns: {missing}")

    rows = []
    trade_rows = []
    for d, day in scored_df.groupby("date", sort=True):
        day = day.copy()
        day["pred_proba"] = pd.to_numeric(day["pred_proba"], errors="coerce")
        day["risk_norm"] = pd.to_numeric(day["risk_norm"], errors="coerce").fillna(0.5).clip(0, 1)
        for c in EXECUTION_PRICE_COLS:
            day[c] = pd.to_numeric(day[c], errors="coerce")
        day = day.dropna(subset=["pred_proba"] + EXECUTION_PRICE_COLS)
        day = day[day["entry_price_opening"] > 0].copy()
        if day.empty:
            rows.append(
                {
                    "date": d,
                    "positions": 0,
                    "gross_return": 0.0,
                    "net_return": 0.0,
                    "capital_start": capital_idr,
                    "capital_end": capital_idr,
                    "effective_threshold": float("nan"),
                }
            )
            continue

        day = day.sort_values("pred_proba", ascending=False).reset_index(drop=True)
        day["pred_rank"] = np.arange(1, len(day) + 1)

        q = float(np.clip(adaptive_threshold_quantile, 0.0, 1.0))
        daily_q = float(day["pred_proba"].quantile(q))
        effective_threshold = (
            max(conviction_min_proba, daily_q)
            if use_adaptive_threshold
            else max(conviction_min_proba, policy.p_cut)
        )

        # High-conviction execution: top-k daily candidates above daily effective threshold.
        if conviction_top_k > 0:
            day = day[day["pred_rank"] <= conviction_top_k].copy()
        day = day[day["pred_proba"] >= effective_threshold].copy()
        if day.empty:
            rows.append(
                {
                    "date": d,
                    "positions": 0,
                    "gross_return": 0.0,
                    "net_return": 0.0,
                    "capital_start": capital_idr,
                    "capital_end": capital_idr,
                    "effective_threshold": effective_threshold,
                }
            )
            continue

        if policy.mode == "threshold":
            day["score"] = np.maximum(day["pred_proba"] - effective_threshold, 0.0) * (1.0 - day["risk_norm"])
        elif policy.mode == "rank":
            day["score"] = day["pred_proba"] * (1.0 - day["risk_norm"])
        elif policy.mode == "pure":
            # Pure probability ranking — no risk adjustment, no threshold subtraction.
            # Maximally equivalent to daily top-k by pred_proba.
            day["score"] = day["pred_proba"]
        else:
            raise ValueError(f"Unsupported policy mode: {policy.mode}")

        day = day[day["score"] > 0].sort_values("score", ascending=False).head(policy.max_positions)
        if day.empty:
            rows.append(
                {
                    "date": d,
                    "positions": 0,
                    "gross_return": 0.0,
                    "net_return": 0.0,
                    "capital_start": capital_idr,
                    "capital_end": capital_idr,
                    "effective_threshold": effective_threshold,
                }
            )
            continue

        weights = capped_weights(day["score"].to_numpy(), policy.max_weight_per_name)
        day["weight"] = weights
        day["allocation_idr"] = capital_idr * day["weight"]
        day = day[day["allocation_idr"] >= min_alloc_idr].copy()
        if day.empty:
            rows.append(
                {
                    "date": d,
                    "positions": 0,
                    "gross_return": 0.0,
                    "net_return": 0.0,
                    "capital_start": capital_idr,
                    "capital_end": capital_idr,
                }
            )
            continue

        # Intra-bar pessimistic fill:
        # If SL touched any time during the bar window, assume SL hit first.
        high_pct = (day["high_to_cutoff"] - day["entry_price_opening"]) / day["entry_price_opening"]
        low_pct = (day["low_to_cutoff"] - day["entry_price_opening"]) / day["entry_price_opening"]
        close_pct = (day["close_to_cutoff"] - day["entry_price_opening"]) / day["entry_price_opening"]

        day["trade_gross_return"] = np.where(
            low_pct <= sl_pct,
            sl_pct,
            np.where(high_pct >= tp_pct, tp_pct, close_pct),
        )
        day["trade_net_return"] = day["trade_gross_return"] - total_cost_frac
        day["weighted_gross"] = day["weight"] * day["trade_gross_return"]
        day["weighted_net"] = day["weight"] * day["trade_net_return"]
        gross_day = float(day["weighted_gross"].sum())
        net_day = float(day["weighted_net"].sum())

        rows.append(
            {
                "date": d,
                "positions": int(len(day)),
                "gross_return": gross_day,
                "net_return": net_day,
                "capital_start": capital_idr,
                "capital_end": capital_idr * (1.0 + net_day),
                "effective_threshold": effective_threshold,
            }
        )
        trade_rows.append(day)

    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if daily.empty:
        summary = {
            "days": 0,
            "trading_days": 0,
            "mean_daily_net_return": float("nan"),
            "mean_daily_gross_return": float("nan"),
            "net_expectancy_per_trade": float("nan"),
            "win_rate_days": float("nan"),
            "cumulative_net_return": float("nan"),
            "max_drawdown": float("nan"),
            "volatility_daily": float("nan"),
        }
        return daily, summary

    equity = (1.0 + daily["net_return"]).cumprod()
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0

    trades = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
    net_expectancy_trade = float(trades["trade_net_return"].mean()) if not trades.empty else 0.0

    summary = {
        "days": int(len(daily)),
        "trading_days": int((daily["positions"] > 0).sum()),
        "mean_daily_net_return": float(daily["net_return"].mean()),
        "mean_daily_gross_return": float(daily["gross_return"].mean()),
        "mean_effective_threshold": float(pd.to_numeric(daily["effective_threshold"], errors="coerce").mean()),
        "net_expectancy_per_trade": net_expectancy_trade,
        "win_rate_days": float((daily["net_return"] > 0).mean()),
        "cumulative_net_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "volatility_daily": float(daily["net_return"].std(ddof=0)),
    }
    return daily, summary


def build_walkforward_splits(
    days: list[pd.Timestamp],
    folds: int,
    valid_days: int,
    min_train_days: int,
) -> list[tuple[list[pd.Timestamp], list[pd.Timestamp]]]:
    if folds <= 0 or valid_days <= 0:
        return []
    total = len(days)
    tail = folds * valid_days
    if total <= tail + min_train_days:
        return []

    start = total - tail
    splits = []
    for i in range(folds):
        v_start = start + i * valid_days
        v_end = v_start + valid_days
        train_days = days[:v_start]
        valid_days_list = days[v_start:v_end]
        if len(train_days) < min_train_days or not valid_days_list:
            continue
        splits.append((train_days, valid_days_list))
    return splits


def apply_oversampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    ratio: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Oversample positive class to achieve target positive ratio.
    ratio = positive / (positive + negative) after oversampling.
    Uses random duplication of positive samples.
    """
    if ratio <= 0.0:
        return X_train, y_train
    # Compute current positive ratio
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    current_ratio = pos / (pos + neg) if (pos + neg) > 0 else 0.0
    if current_ratio >= ratio:
        # Already meets ratio
        return X_train, y_train
    # Determine target number of positive samples
    target_pos = int(neg * ratio / (1 - ratio)) if ratio < 1.0 else neg  # avoid division by zero
    # Number of additional positive samples needed
    needed = target_pos - pos
    if needed <= 0:
        return X_train, y_train
    # Get indices of positive samples
    pos_idx = y_train[y_train == 1].index
    # Randomly sample with replacement
    rng = np.random.RandomState(random_state)
    extra_idx = rng.choice(pos_idx, size=needed, replace=True)
    # Concatenate
    X_extra = X_train.loc[extra_idx].reset_index(drop=True)
    y_extra = y_train.loc[extra_idx].reset_index(drop=True)
    X_oversampled = pd.concat([X_train, X_extra], ignore_index=True)
    y_oversampled = pd.concat([y_train, y_extra], ignore_index=True)
    return X_oversampled, y_oversampled


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    args: argparse.Namespace,
) -> lgb.LGBMClassifier:
    # Apply oversampling if requested
    if args.oversample_ratio > 0.0:
        X_train, y_train = apply_oversampling(
            X_train, y_train, args.oversample_ratio, args.random_state
        )
    
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    auto_spw = float(neg / max(pos, 1))
    scale_pos_weight = float(args.scale_pos_weight if args.scale_pos_weight > 0 else auto_spw)
    # Apply multiplier
    if args.pos_weight_multiplier != 1.0:
        scale_pos_weight *= args.pos_weight_multiplier

    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_data_in_leaf=args.min_data_in_leaf,
        feature_fraction=args.feature_fraction,
        bagging_fraction=args.bagging_fraction,
        bagging_freq=args.bagging_freq,
        lambda_l1=args.lambda_l1,
        lambda_l2=args.lambda_l2,
        min_gain_to_split=args.min_gain_to_split,
        force_col_wise=True,
        verbosity=-1,
        random_state=args.random_state,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=["auc", "average_precision", "binary_logloss"],
        callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)],
    )
    return model


def fail_with_reason(output_dir: Path, reason: str, details: dict | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"status": "FAIL", "reason": reason, "details": details or {}}
    path = output_dir / "metrics.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"MODEL_TRAINING_STATUS=FAIL:{reason}")
    print(f"[FailArtifact] {path}")
    sys.exit(1)


def main() -> None:
    args = parse_args()
    topk_list = parse_int_list(args.topk_list)
    p_grid = parse_float_list(args.p_cut_grid)
    policy_modes = parse_mode_list(args.policy_modes)
    k_grid = parse_int_list(args.max_positions_grid)
    w_grid = parse_float_list(args.max_weight_grid)

    if not args.training_path.exists():
        fail_with_reason(args.output_dir, "MISSING_TRAINING_DATA", {"path": str(args.training_path)})

    df = pd.read_parquet(args.training_path)
    if TARGET_COL not in df.columns:
        fail_with_reason(args.output_dir, "MISSING_TARGET_COLUMN", {"target": TARGET_COL})
    missing_exec_cols = [c for c in EXECUTION_PRICE_COLS if c not in df.columns]
    if missing_exec_cols:
        fail_with_reason(args.output_dir, "MISSING_EXECUTION_COLUMNS", {"columns": missing_exec_cols})
    if args.tp_pct <= 0:
        fail_with_reason(args.output_dir, "INVALID_TP_PCT", {"tp_pct": args.tp_pct})
    if args.sl_pct >= 0:
        fail_with_reason(args.output_dir, "INVALID_SL_PCT", {"sl_pct": args.sl_pct})

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date", "ticker", TARGET_COL]).copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    unique_days = sorted(df["date"].unique().tolist())
    if len(unique_days) < args.min_trading_days:
        fail_with_reason(
            args.output_dir,
            "FAIL_MIN_HISTORY",
            {"trading_days": len(unique_days), "min_required": args.min_trading_days},
        )

    if len(unique_days) <= args.oot_valid_days + 10:
        fail_with_reason(
            args.output_dir,
            "INSUFFICIENT_DAYS_FOR_OOT",
            {"trading_days": len(unique_days), "oot_valid_days": args.oot_valid_days},
        )

    pre_oot_days = unique_days[:-args.oot_valid_days]
    oot_days = unique_days[-args.oot_valid_days :]
    pre_oot_df = df[df["date"].isin(pre_oot_days)].copy()
    oot_df = df[df["date"].isin(oot_days)].copy()

    ranked_candidates, feature_source = load_ranked_feature_candidates(
        path=args.feature_importance_path,
        top_n=args.feature_prune_top_n,
    )
    feature_cols = choose_feature_columns(df, ranked_candidates=ranked_candidates)
    if not feature_cols:
        fail_with_reason(args.output_dir, "NO_USABLE_FEATURE_COLUMNS")

    leakage_in_features = sorted(set(feature_cols) & OUTCOME_COLS)
    if leakage_in_features:
        fail_with_reason(args.output_dir, "LEAKAGE_FEATURES_DETECTED", {"columns": leakage_in_features})

    X_pre = pre_oot_df[feature_cols]
    all_null = [c for c in feature_cols if X_pre[c].isna().all()]
    constant = [c for c in feature_cols if X_pre[c].nunique(dropna=True) <= 1]
    duplicate = X_pre.T.duplicated()
    duplicate_cols = sorted(X_pre.columns[duplicate].tolist())
    drop_cols = sorted(set(all_null + constant + duplicate_cols))
    backtest_cols = list(EXECUTION_PRICE_COLS)
    if "close_return_to_cutoff" in df.columns:
        backtest_cols.append("close_return_to_cutoff")
    if drop_cols:
        feature_cols = [c for c in feature_cols if c not in drop_cols]
        base_cols = ["date", "ticker", TARGET_COL] + backtest_cols
        pre_oot_df = pre_oot_df[base_cols + feature_cols].copy()
        oot_df = oot_df[base_cols + feature_cols].copy()

    # Walk-forward CV for robust policy search.
    wf_splits = build_walkforward_splits(
        days=pre_oot_days,
        folds=args.walkforward_folds,
        valid_days=args.walkforward_valid_days,
        min_train_days=args.min_trading_days,
    )
    if not wf_splits:
        fail_with_reason(
            args.output_dir,
            "INSUFFICIENT_DAYS_FOR_WALKFORWARD",
            {
                "pre_oot_days": len(pre_oot_days),
                "folds": args.walkforward_folds,
                "fold_valid_days": args.walkforward_valid_days,
                "min_train_days": args.min_trading_days,
            },
        )

    print(f"[Init] training_path={args.training_path}")
    print(f"[Init] rows={len(df):,}, days={len(unique_days)}, date={unique_days[0].date()} -> {unique_days[-1].date()}")
    print(f"[Split] pre_oot_days={len(pre_oot_days)}, oot_days={len(oot_days)}")
    print(
        f"[Feature] selected={len(feature_cols)}, dropped={len(drop_cols)}, "
        f"source={feature_source}, prune_top_n={args.feature_prune_top_n}"
    )
    print(f"[WalkForward] folds={len(wf_splits)}, each_valid_days={args.walkforward_valid_days}")

    wf_records = []
    wf_metrics = []
    for idx, (tr_days, va_days) in enumerate(wf_splits, start=1):
        tr = pre_oot_df[pre_oot_df["date"].isin(tr_days)].copy()
        va = pre_oot_df[pre_oot_df["date"].isin(va_days)].copy()
        X_tr, y_tr = tr[feature_cols], tr[TARGET_COL]
        X_va, y_va = va[feature_cols], va[TARGET_COL]

        model_fold = train_lgbm(X_tr, y_tr, X_va, y_va, args)
        proba_va = model_fold.predict_proba(X_va)[:, 1]
        fold_metric = binary_metrics(y_va, proba_va)
        wf_metrics.append(
            {
                "fold": idx,
                "train_days": len(tr_days),
                "valid_days": len(va_days),
                "valid_rows": int(len(va)),
                **fold_metric,
            }
        )

        va_rec = va[["date", "ticker", TARGET_COL] + backtest_cols].copy()
        va_rec["pred_proba"] = proba_va
        va_rec["risk_norm"] = build_risk_proxy(va[feature_cols])
        wf_records.append(va_rec)
        print(
            f"[Fold {idx}] valid_rows={len(va):,}, aucpr={fold_metric['aucpr']:.6f}, "
            f"auc={fold_metric['auc']:.6f}, logloss={fold_metric['logloss']:.6f}"
        )

    wf_pred = pd.concat(wf_records, ignore_index=True).sort_values(["date", "pred_proba"], ascending=[True, False])
    total_cost_frac = (args.buy_cost_bps + args.sell_cost_bps + 2.0 * args.slippage_bps_per_side) / 10_000.0

    # Policy search on walk-forward validation predictions.
    best_policy = None
    best_summary = None
    best_daily = None
    for mode in policy_modes:
        cut_values = p_grid if mode == "threshold" else [0.0]
        for p_cut in cut_values:
            for k in k_grid:
                for w in w_grid:
                    policy = PolicyConfig(
                        mode=mode,
                        p_cut=float(p_cut),
                        max_positions=int(k),
                        max_weight_per_name=float(w),
                    )
                    daily, summary = simulate_portfolio(
                        scored_df=wf_pred,
                        policy=policy,
                        capital_idr=args.capital_idr,
                        min_alloc_idr=args.min_allocation_idr,
                        total_cost_frac=total_cost_frac,
                        conviction_top_k=args.conviction_top_k,
                        conviction_min_proba=args.conviction_min_proba,
                        use_adaptive_threshold=args.use_adaptive_threshold,
                        adaptive_threshold_quantile=args.adaptive_threshold_quantile,
                        tp_pct=args.tp_pct,
                        sl_pct=args.sl_pct,
                    )
                    if summary["trading_days"] < args.min_policy_trading_days:
                        continue
                    score = summary["mean_daily_net_return"]
                    if math.isnan(score):
                        continue
                    if best_summary is None:
                        best_policy, best_summary, best_daily = policy, summary, daily
                        continue
                    cur = (
                        score,
                        -abs(summary["max_drawdown"]),
                        summary["trading_days"],
                    )
                    prev = (
                        best_summary["mean_daily_net_return"],
                        -abs(best_summary["max_drawdown"]),
                        best_summary["trading_days"],
                    )
                    if cur > prev:
                        best_policy, best_summary, best_daily = policy, summary, daily

    if best_policy is None or best_summary is None:
        # Fallback to deterministic high-conviction policy, still allowing full metrics reporting.
        fallback_max_positions = max(1, min(3, args.conviction_top_k))
        fallback_policy = PolicyConfig(
            mode="threshold",
            p_cut=args.conviction_min_proba,
            max_positions=fallback_max_positions,
            max_weight_per_name=max(w_grid),
        )
        best_daily, best_summary = simulate_portfolio(
            scored_df=wf_pred,
            policy=fallback_policy,
            capital_idr=args.capital_idr,
            min_alloc_idr=args.min_allocation_idr,
            total_cost_frac=total_cost_frac,
            conviction_top_k=args.conviction_top_k,
            conviction_min_proba=args.conviction_min_proba,
            use_adaptive_threshold=args.use_adaptive_threshold,
            adaptive_threshold_quantile=args.adaptive_threshold_quantile,
            tp_pct=args.tp_pct,
            sl_pct=args.sl_pct,
        )
        best_policy = fallback_policy

    print(
        "[PolicySelect] mode={}, p_cut={:.4f}, max_positions={}, max_weight={:.2f}, cv_mean_daily_net={:.6f}, "
        "cv_cum_net={:.6f}".format(
            best_policy.mode,
            best_policy.p_cut,
            best_policy.max_positions,
            best_policy.max_weight_per_name,
            best_summary["mean_daily_net_return"],
            best_summary["cumulative_net_return"],
        )
    )

    # Final model training on pre-OOT with tail for early stopping.
    if len(pre_oot_days) <= args.earlystop_valid_days + args.min_trading_days:
        fail_with_reason(
            args.output_dir,
            "INSUFFICIENT_DAYS_FOR_FINAL_EARLYSTOP",
            {
                "pre_oot_days": len(pre_oot_days),
                "earlystop_valid_days": args.earlystop_valid_days,
                "min_train_days": args.min_trading_days,
            },
        )

    es_days = pre_oot_days[-args.earlystop_valid_days :]
    core_days = pre_oot_days[: -args.earlystop_valid_days]
    core_df = pre_oot_df[pre_oot_df["date"].isin(core_days)].copy()
    es_df = pre_oot_df[pre_oot_df["date"].isin(es_days)].copy()

    final_model = train_lgbm(
        X_train=core_df[feature_cols],
        y_train=core_df[TARGET_COL],
        X_valid=es_df[feature_cols],
        y_valid=es_df[TARGET_COL],
        args=args,
    )

    train_prob = final_model.predict_proba(core_df[feature_cols])[:, 1]
    oot_prob = final_model.predict_proba(oot_df[feature_cols])[:, 1]

    train_metrics = binary_metrics(core_df[TARGET_COL], train_prob)
    oot_metrics = binary_metrics(oot_df[TARGET_COL], oot_prob)
    overfit_gap = {
        "auc_train_minus_oot": float(train_metrics["auc"] - oot_metrics["auc"]),
        "aucpr_train_minus_oot": float(train_metrics["aucpr"] - oot_metrics["aucpr"]),
    }

    oot_pred = oot_df[["date", "ticker", TARGET_COL] + backtest_cols].copy()
    oot_pred["pred_proba"] = oot_prob
    oot_pred["risk_norm"] = build_risk_proxy(oot_df[feature_cols])
    oot_pred = oot_pred.sort_values(["date", "pred_proba"], ascending=[True, False]).reset_index(drop=True)
    oot_pred["rank_daily"] = oot_pred.groupby("date", sort=False)["pred_proba"].rank(method="first", ascending=False)

    oot_topk = daily_topk_metrics(oot_pred, k_list=topk_list)
    oot_daily_port, oot_port_summary = simulate_portfolio(
        scored_df=oot_pred,
        policy=best_policy,
        capital_idr=args.capital_idr,
        min_alloc_idr=args.min_allocation_idr,
        total_cost_frac=total_cost_frac,
        conviction_top_k=args.conviction_top_k,
        conviction_min_proba=args.conviction_min_proba,
        use_adaptive_threshold=args.use_adaptive_threshold,
        adaptive_threshold_quantile=args.adaptive_threshold_quantile,
        tp_pct=args.tp_pct,
        sl_pct=args.sl_pct,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model_lightgbm_opening_tp3.txt"
    final_model.booster_.save_model(str(model_path))

    fi = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_gain": final_model.booster_.feature_importance(importance_type="gain"),
            "importance_split": final_model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    fi_path = args.output_dir / "feature_importance.csv"
    fi.to_csv(fi_path, index=False)

    oot_pred_path = args.output_dir / "valid_predictions.parquet"
    oot_pred.to_parquet(oot_pred_path, index=False)

    portfolio_daily_path = args.output_dir / "portfolio_daily.parquet"
    oot_daily_port.to_parquet(portfolio_daily_path, index=False)

    wf_metrics_df = pd.DataFrame(wf_metrics)
    wf_metrics_path = args.output_dir / "walkforward_metrics.csv"
    wf_metrics_df.to_csv(wf_metrics_path, index=False)

    status = "PASS" if oot_port_summary["mean_daily_net_return"] > 0 else "FAIL:NEGATIVE_OOT_EXPECTANCY"
    metrics_payload = {
        "status": status,
        "northstar": {
            "capital_idr": args.capital_idr,
            "entry_time": "09:00",
            "exit_time": "10:00",
            "target": "positive_net_expectancy_after_costs",
        },
        "data": {
            "training_path": str(args.training_path),
            "rows_total": int(len(df)),
            "trading_days_total": int(len(unique_days)),
            "date_min": str(unique_days[0].date()),
            "date_max": str(unique_days[-1].date()),
            "pre_oot_days": int(len(pre_oot_days)),
            "oot_days": int(len(oot_days)),
        },
        "hard_rules": {
            "min_history_required_days": int(args.min_trading_days),
            "min_history_observed_days": int(len(unique_days)),
            "no_random_split": True,
            "no_lookahead_features": True,
        },
        "features": {
            "selected_count": int(len(feature_cols)),
            "dropped_columns": drop_cols,
            "feature_prune_top_n": int(args.feature_prune_top_n),
            "feature_source": feature_source,
            "ranked_candidates_count": int(len(ranked_candidates)),
        },
        "walkforward": {
            "folds": int(len(wf_splits)),
            "fold_valid_days": int(args.walkforward_valid_days),
            "metrics": wf_metrics,
            "policy_selected": {
                "mode": best_policy.mode,
                "p_cut": best_policy.p_cut,
                "max_positions": best_policy.max_positions,
                "max_weight_per_name": best_policy.max_weight_per_name,
            },
            "policy_cv_summary": best_summary,
        },
        "model": {
            "type": "lightgbm.LGBMClassifier",
            "best_iteration": int(final_model.best_iteration_),
            "params": {
                "n_estimators": args.n_estimators,
                "learning_rate": args.learning_rate,
                "num_leaves": args.num_leaves,
                "max_depth": args.max_depth,
                "min_data_in_leaf": args.min_data_in_leaf,
                "feature_fraction": args.feature_fraction,
                "bagging_fraction": args.bagging_fraction,
                "bagging_freq": args.bagging_freq,
                "lambda_l1": args.lambda_l1,
                "lambda_l2": args.lambda_l2,
                "min_gain_to_split": args.min_gain_to_split,
                "scale_pos_weight": args.scale_pos_weight if args.scale_pos_weight > 0 else "auto",
                "early_stopping_rounds": args.early_stopping_rounds,
            },
        },
        "metrics": {
            "train_core": train_metrics,
            "oot_valid": oot_metrics,
            "oot_daily_topk": oot_topk,
            "overfit_gap": overfit_gap,
            "portfolio_oot": oot_port_summary,
        },
        "cost_assumptions": {
            "buy_cost_bps": args.buy_cost_bps,
            "sell_cost_bps": args.sell_cost_bps,
            "slippage_bps_per_side": args.slippage_bps_per_side,
            "total_cost_fraction_per_roundtrip": total_cost_frac,
        },
        "execution": {
            "conviction_top_k": int(args.conviction_top_k),
            "conviction_min_proba": float(args.conviction_min_proba),
            "use_adaptive_threshold": bool(args.use_adaptive_threshold),
            "adaptive_threshold_quantile": float(args.adaptive_threshold_quantile),
            "tp_pct": float(args.tp_pct),
            "sl_pct": float(args.sl_pct),
            "mean_effective_threshold_oot": float(oot_port_summary.get("mean_effective_threshold", float("nan"))),
        },
        "artifacts": {
            "model_path": str(model_path),
            "feature_importance_path": str(fi_path),
            "valid_predictions_path": str(oot_pred_path),
            "portfolio_daily_path": str(portfolio_daily_path),
            "walkforward_metrics_path": str(wf_metrics_path),
        },
    }

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(f"[OOT] aucpr={oot_metrics['aucpr']:.6f}, auc={oot_metrics['auc']:.6f}, logloss={oot_metrics['logloss']:.6f}")
    print(
        f"[OOT Portfolio] mean_daily_net={oot_port_summary['mean_daily_net_return']:.6f}, "
        f"cum_net={oot_port_summary['cumulative_net_return']:.6f}, max_dd={oot_port_summary['max_drawdown']:.6f}, "
        f"mean_effective_threshold={oot_port_summary.get('mean_effective_threshold', float('nan')):.6f}"
    )
    print(f"[OverfitGap] auc_gap={overfit_gap['auc_train_minus_oot']:.6f}, aucpr_gap={overfit_gap['aucpr_train_minus_oot']:.6f}")
    print(f"[Done] metrics={metrics_path}")
    print(f"MODEL_TRAINING_STATUS={status}")


if __name__ == "__main__":
    main()
