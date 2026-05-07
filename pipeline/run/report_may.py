import pandas as pd
import lightgbm as lgb
import numpy as np
import duckdb
from pathlib import Path
import sys

def report_may_performance():
    repo_root = Path(__file__).resolve().parents[2]
    model_path = repo_root / "model" / "BSJP" / "bsjp_v19d_close10_preclose14_orb_md100_l21.5" / "model_lightgbm_opening_tp3.txt"
    db_path = repo_root / "inferences" / "bsjp" / "db" / "inference.duckdb"
    l2_path = repo_root / "data" / "Level_2_Datamart" / "training_datamart_bsjp_overnight.parquet"
    p14_path = repo_root / "data" / "Level_1_Features" / "modules" / "preclose14_features.parquet"
    ov_path = repo_root / "data" / "Level_1_Features" / "modules" / "overnight_history_features.parquet"
    yf_1h_path = repo_root / "data" / "Level_0_Raw" / "yfinance_1h.parquet"

    # 1. Load Model
    bst = lgb.Booster(model_file=str(model_path))
    fnames = bst.feature_name()

    # 2. Prepare Data for May
    print("Loading feature modules...")
    p14 = pd.read_parquet(p14_path)
    ov = pd.read_parquet(ov_path)
    # Join modules to get complete feature set
    m = p14.merge(ov, on=['date', 'ticker'], how='inner')
    m['date_str'] = pd.to_datetime(m['date']).dt.strftime('%Y-%m-%d')
    m['ticker'] = m['ticker'].str.replace('.JK', '', regex=False)
    
    # 3. Get Prices for PnL
    print("Loading raw prices for PnL calculation...")
    yf = pd.read_parquet(yf_1h_path)
    yf['dt_wib'] = pd.to_datetime(yf['datetime'], utc=True).dt.tz_convert('Asia/Jakarta')
    yf['date_str'] = yf['dt_wib'].dt.strftime('%Y-%m-%d')
    yf['ticker'] = yf['ticker'].str.replace('.JK', '', regex=False)
    
    # Entry: May 4 Close @ 15:xx
    entry_df = yf[(yf['date_str'] == '2026-05-04') & (yf['dt_wib'].dt.hour == 15)].groupby('ticker')['close'].last().reset_index(name='entry_price')
    # Exit: May 5 Open @ 09:xx
    exit_df = yf[(yf['date_str'] == '2026-05-05') & (yf['dt_wib'].dt.hour == 9)].groupby('ticker')['open'].first().reset_index(name='exit_price')
    
    # 4. Predict & Performance for May 4
    may4 = m[m['date_str'] == '2026-05-04'].copy()
    if may4.empty:
        print("❌ No data for May 4")
    else:
        X = may4[fnames].fillna(0).astype(np.float32)
        may4['proba'] = bst.predict(X)
        top3 = may4.sort_values('proba', ascending=False).head(3).copy()
        
        # Merge prices
        top3 = top3.merge(entry_df, on='ticker', how='left')
        top3 = top3.merge(exit_df, on='ticker', how='left')
        top3['return'] = (top3['exit_price'] - top3['entry_price']) / top3['entry_price']
        
        print("\n=== MAY 4 2026 PERFORMANCE (Exit May 5 Open) ===")
        print(top3[['ticker', 'proba', 'entry_price', 'exit_price', 'return']].to_string(index=False))
        print(f"\nAverage Return May 4: {top3['return'].mean():.2%}")

    # 5. Top 3 for May 5 (Today)
    may5 = m[m['date_str'] == '2026-05-05'].copy()
    if may5.empty:
        print("\n❌ No data for May 5")
    else:
        X5 = may5[fnames].fillna(0).astype(np.float32)
        may5['proba'] = bst.predict(X5)
        top3_5 = may5.sort_values('proba', ascending=False).head(3).copy()
        print("\n=== MAY 5 2026 RECOMMENDATIONS (Today) ===")
        print(top3_5[['ticker', 'proba']].to_string(index=False))

if __name__ == "__main__":
    report_may_performance()
