import pandas as pd
import numpy as np
from pathlib import Path
import time

# --- Config ---
# Path relative to script: edges/bsjp_overnight_sl2/scripts/ -> edges -> idx/
IDX_DIR = Path(__file__).parent.parent.parent.parent
L0_DIR = IDX_DIR / "data" / "Level_0_Raw"
L1_DIR = IDX_DIR / "data" / "Level_1_Features" / "modules"
INPUT_PATH = L0_DIR / "yfinance_1h.parquet"
OUTPUT_PATH = L1_DIR / "session_intensity_features.parquet"

def generate():
    start_time = time.time()
    print(f"Reading {INPUT_PATH}...")
    df = pd.read_parquet(INPUT_PATH)
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    
    # Vectorized Intensity Calculation (Buying Power Proxy)
    epsilon = 1e-6
    df['intensity'] = df['volume'] * (df['close'] - df['low']) / (df['high'] - df['low'] + epsilon)
    
    print("Computing ANTI-LOOKAHEAD session features...")
    print("NOTE: Hour 14:00 is EXCLUDED as it contains data up to 15:00 (unknown at decision time).")
    
    # Create masks for SAFE sessions
    # S1: 09:00, 10:00, 11:00 (known at 12:00)
    # S2_SAFE: 13:00 (known at 14:00)
    # We EXCLUDE 14:00 (which contains 14:00 - 15:00)
    s1_mask = df['hour'].isin([9, 10, 11])
    s2_safe_mask = df['hour'] == 13
    early_s1_mask = df['hour'].isin([9, 10])
    
    # Pre-calculate aggregates
    df['int_s1'] = np.where(s1_mask, df['intensity'], 0)
    df['vol_s1'] = np.where(s1_mask, df['volume'], 0)
    df['int_s2_safe'] = np.where(s2_safe_mask, df['intensity'], 0)
    df['vol_s2_safe'] = np.where(s2_safe_mask, df['volume'], 0)
    df['int_early_s1'] = np.where(early_s1_mask, df['intensity'], 0)
    
    # Groupby and sum
    features = df.groupby(['date', 'ticker']).agg({
        'int_s1': 'sum',
        'int_s2_safe': 'sum',
        'int_early_s1': 'sum',
        'vol_s1': 'sum',
        'vol_s2_safe': 'sum'
    }).reset_index()
    
    # Rename and compute ratios
    features.rename(columns={
        'int_s1': 'intensity_s1_total',
        'int_s2_safe': 'intensity_s2_until_14h', # Known at 14:00
        'int_early_s1': 'intensity_early_s1'
    }, inplace=True)
    
    # Key Alpha Feature: Does intensity explode in the first hour of Session 2?
    features['intensity_ratio_s2_s1'] = features['intensity_s2_until_14h'] / (features['intensity_s1_total'] + epsilon)
    features['vol_ratio_s2_s1'] = features['vol_s2_safe'] / (features['vol_s1'] + epsilon)
    
    # Cleanup columns
    features = features[['date', 'ticker', 'intensity_s1_total', 'intensity_s2_until_14h', 
                         'intensity_early_s1', 'intensity_ratio_s2_s1', 'vol_ratio_s2_s1']]
    
    # Normalize ticker keys
    features['ticker'] = features['ticker'].str.replace('.JK', '', regex=False)
    features['date'] = pd.to_datetime(features['date'])
    
    # Ensure float32 for model consistency
    feat_cols = [c for c in features.columns if c not in ['date', 'ticker']]
    features[feat_cols] = features[feat_cols].astype('float32')
    
    print(f"Saving to {OUTPUT_PATH}...")
    features.to_parquet(OUTPUT_PATH, index=False)
    print(f"Done in {time.time() - start_time:.2f}s. Shape: {features.shape}")

if __name__ == "__main__":
    generate()
