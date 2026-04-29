#!/usr/bin/env python3
"""
Analyze feature importance stability across model versions.
"""
import pandas as pd
import numpy as np
from pathlib import Path

MODEL_DIR = Path("idx/model")
VERSIONS = [
    "bpjs_lgbm_opening_tp3_v9",
    "bpjs_lgbm_opening_tp3_v10",
    "bpjs_lgbm_opening_tp3_v11",
    "bpjs_lgbm_opening_tp3_v11_full",
    "bpjs_lgbm_opening_tp3_v12_sniper",
]

def load_top_features(version, top_n=20):
    path = MODEL_DIR / version / "feature_importance.csv"
    if not path.exists():
        print(f"  Skipping {version} (file not found)")
        return []
    df = pd.read_csv(path)
    # Sort by gain importance descending
    df = df.sort_values("importance_gain", ascending=False).reset_index(drop=True)
    top = df.head(top_n)["feature"].tolist()
    return top

def main():
    print("Feature Importance Stability Analysis")
    print("=" * 50)
    top_features = {}
    for v in VERSIONS:
        top = load_top_features(v)
        top_features[v] = set(top)
        print(f"{v}: {len(top)} features")
        if top:
            print("  Top 5:", ", ".join(top[:5]))
    
    # Compute pairwise Jaccard similarity
    versions = list(top_features.keys())
    n = len(versions)
    for i in range(n):
        for j in range(i+1, n):
            set_i = top_features[versions[i]]
            set_j = top_features[versions[j]]
            if not set_i or not set_j:
                continue
            intersection = set_i.intersection(set_j)
            union = set_i.union(set_j)
            jaccard = len(intersection) / len(union) if union else 0
            print(f"Jaccard({versions[i]}, {versions[j]}) = {jaccard:.3f}")
    
    # Intersection across all versions
    common = set.intersection(*[top_features[v] for v in versions if top_features[v]])
    print(f"\nFeatures present in all {len(versions)} versions ({len(common)}):")
    for feat in sorted(common):
        print(f"  - {feat}")
    
    # Compute rank correlation (simplified)
    # For each feature, compute its average rank across versions where it appears
    feature_ranks = {}
    for v in VERSIONS:
        path = MODEL_DIR / v / "feature_importance.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df.sort_values("importance_gain", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        for _, row in df.iterrows():
            feat = row["feature"]
            feature_ranks.setdefault(feat, []).append(row["rank"])
    
    avg_ranks = {}
    for feat, ranks in feature_ranks.items():
        avg_ranks[feat] = np.mean(ranks)
    
    top_avg = sorted(avg_ranks.items(), key=lambda x: x[1])[:20]
    print("\nTop 20 features by average rank:")
    for feat, avg in top_avg:
        print(f"  {feat}: {avg:.1f} (appeared in {len(feature_ranks[feat])} versions)")

if __name__ == "__main__":
    main()