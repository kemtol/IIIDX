import pandas as pd
import numpy as np
from pathlib import Path

IDX_DIR = Path("/home/kemal/idx")
L0_PATH = IDX_DIR / "data/Level_0_Raw/broksum_bybroker.parquet"
OUTPUT_PATH = IDX_DIR / "data/Level_1_Features/modules/forensic_v2_features.parquet"

def build_forensic_v2():
    print("--- Building Forensic Features V2 (Micro Momentum & Decay Inventory) ---")
    df = pd.read_parquet(L0_PATH)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    
    # 1. Decay Inventory
    df_sum = df.groupby(['date', 'stock_code'])['net_val'].sum().reset_index()
    df_sum = df_sum.sort_values(['stock_code', 'date'])
    df_sum['f2_inventory_decay'] = df_sum.groupby('stock_code')['net_val'].transform(lambda x: x.ewm(alpha=0.3).sum())
    
    # 2. Absorption Index
    yf = pd.read_parquet(IDX_DIR / "data/Level_0_Raw/yfinance_daily.parquet")
    yf['date'] = pd.to_datetime(yf['date']).dt.normalize()
    yf['ticker'] = yf['ticker'].str.replace('.JK', '', regex=False)
    
    df_sum = df_sum.merge(yf[['date', 'ticker', 'close', 'open']], left_on=['date', 'stock_code'], right_on=['date', 'ticker'], how='left')
    df_sum['day_ret'] = (df_sum['close'] - df_sum['open']) / df_sum['open'].replace(0, np.nan)
    df_sum['f2_absorption_ratio'] = df_sum['net_val'] / (df_sum['day_ret'].abs() + 0.001)
    
    # 3. Last Hour Momentum
    yf1h = pd.read_parquet(IDX_DIR / "data/Level_0_Raw/yfinance_1h.parquet")
    yf1h['datetime'] = pd.to_datetime(yf1h['datetime']).dt.tz_localize(None)
    yf1h['date'] = yf1h['datetime'].dt.normalize()
    yf1h['ticker'] = yf1h['ticker'].str.replace('.JK', '', regex=False)
    yf1h['hour'] = yf1h['datetime'].dt.hour
    
    morning_vol = yf1h[yf1h['hour'] < 13].groupby(['date', 'ticker'])['volume'].mean().reset_index(name='avg_morning_vol')
    afternoon_vol = yf1h[yf1h['hour'] == 14].groupby(['date', 'ticker'])['volume'].sum().reset_index(name='vol_14h')
    
    vol_surge = afternoon_vol.merge(morning_vol, on=['date', 'ticker'], how='left')
    vol_surge['f2_vol_surge_14h'] = vol_surge['vol_14h'] / vol_surge['avg_morning_vol'].replace(0, np.nan)
    
    # Final assembly
    forensic_v2 = df_sum[['date', 'stock_code', 'f2_inventory_decay', 'f2_absorption_ratio']].rename(columns={'stock_code':'ticker'})
    forensic_v2 = forensic_v2.merge(vol_surge[['date', 'ticker', 'f2_vol_surge_14h']], on=['date', 'ticker'], how='left')
    
    forensic_v2.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {len(forensic_v2)} rows to {OUTPUT_PATH}")

if __name__ == "__main__":
    build_forensic_v2()
