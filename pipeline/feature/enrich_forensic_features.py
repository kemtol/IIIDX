import pandas as pd
import numpy as np
from pathlib import Path

IDX_DIR = Path("/home/kemal/idx")
L0_PATH = IDX_DIR / "data/Level_0_Raw/broksum_bybroker.parquet"
OUTPUT_PATH = IDX_DIR / "data/Level_1_Features/modules/forensic_features.parquet"

def build_forensic_features():
    print("--- Building Forensic Features (Inventory & Aggression) ---")
    df = pd.read_parquet(L0_PATH)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    
    # 1. Aggression Index (Avg Buy Price vs Prev Close)
    # Note: We need prev close from yfinance_daily
    yf = pd.read_parquet(IDX_DIR / "data/Level_0_Raw/yfinance_daily.parquet")
    yf['date'] = pd.to_datetime(yf['date']).dt.normalize()
    yf['ticker'] = yf['ticker'].str.replace('.JK', '', regex=False)
    yf = yf.sort_values(['ticker', 'date'])
    yf['prev_close'] = yf.groupby('ticker')['close'].shift(1)
    
    df = df.merge(yf[['date', 'ticker', 'prev_close']], left_on=['date', 'stock_code'], right_on=['date', 'ticker'], how='left')
    
    # aggression = (avg_buy_price - prev_close) / prev_close
    df['broker_aggression'] = (df['avg_buy_price'] - df['prev_close']) / df['prev_close'].replace(0, np.nan)
    
    # 2. Broker Concentration (HHI)
    # HHI = sum( (broker_net_buy / total_net_buy)^2 )
    ticker_net = df.groupby(['date', 'ticker'])['net_val'].transform('sum')
    df['net_share'] = df['net_val'] / ticker_net.replace(0, np.nan)
    df['net_share_sq'] = df['net_share']**2
    
    # 3. Aggregation
    forensic = df.groupby(['date', 'ticker']).agg(
        forensic_hhi_index=('net_share_sq', 'sum'),
        forensic_max_aggression=('broker_aggression', 'max'),
        forensic_mean_aggression=('broker_aggression', 'mean'),
        forensic_top_broker_share=('net_share', 'max')
    ).reset_index()
    
    # 4. Inventory (Rolling 10d Net Buy sum)
    # This is a bit slow, let's do simple rolling sum per ticker
    df_sorted = df.groupby(['date', 'ticker'])['net_val'].sum().reset_index()
    df_sorted = df_sorted.sort_values(['ticker', 'date'])
    df_sorted['forensic_inventory_10d'] = df_sorted.groupby('ticker')['net_val'].transform(lambda x: x.rolling(10, min_periods=1).sum())
    
    forensic = forensic.merge(df_sorted[['date', 'ticker', 'forensic_inventory_10d']], on=['date', 'ticker'], how='left')
    
    forensic.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {len(forensic)} rows to {OUTPUT_PATH}")

if __name__ == "__main__":
    build_forensic_features()
