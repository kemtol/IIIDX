#!/bin/bash
# Example script to run v8 experiment
# This demonstrates how configurable the system is

cd "$(dirname "$0")"

# Generate new datamart with latest data
python generate_datamart.py \
    --enable-universe-filter \
    --training-output ../../../data/Level_2_Datamart/training_datamart_bsjp_overnight_v8.parquet

# Train v8 with new configuration
python train_lightgbm.py \
    --training-path ../../../data/Level_2_Datamart/training_datamart_bsjp_overnight_v8.parquet \
    --output-dir ../../../model/BSJP/bsjp_v8 \
    --feature-importance-path ../../../model/BSJP/bsjp_v7/feature_importance.csv \
    --feature-prune-top-n 150 \
    --scale-pos-weight 0.0 \
    --pos-weight-multiplier 0.5 \
    --learning-rate 0.015 \
    --num-leaves 24 \
    --max-depth 4 \
    --min-data-in-leaf 400 \
    --tp-pct 0.01 \
    --sl-pct -0.02
