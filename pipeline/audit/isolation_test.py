import pandas as pd
from pathlib import Path
import shutil
import sys
import os

IDX_DIR = Path(__file__).resolve().parents[2]

def run_isolation_audit(target_date, ticker):
    print(f"--- Running Isolation Audit for {target_date} / {ticker} ---")
    
    # 1. Create Sandbox
    sandbox = IDX_DIR / "_LOG/audit_sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    
    # 2. Simulate "Cutting the Future"
    # We copy yfinance_daily but remove everything >= target_date
    raw_path = IDX_DIR / "data/Level_0_Raw/yfinance_daily.parquet"
    sb_path = sandbox / "yfinance_daily_isolated.parquet"
    
    df = pd.read_parquet(raw_path)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    isolated_df = df[df['date'] < pd.to_datetime(target_date)].copy()
    isolated_df.to_parquet(sb_path, index=False)
    
    print(f"Sandbox L0 created: {len(isolated_df)} rows (original: {len(df)}). Max date: {isolated_df['date'].max()}")
    
    # 3. Load Feature from production datamart
    prod_path = IDX_DIR / "data/Level_1_Features/modules/ara_history_features.parquet"
    prod_mod = pd.read_parquet(prod_path)
    prod_mod['date'] = pd.to_datetime(prod_mod['date']).dt.normalize()
    prod_val = prod_mod[(prod_mod['date'] == pd.to_datetime(target_date)) & (prod_mod['ticker'] == ticker)]['ara_count_5d'].values[0]
    
    # 4. Rebuild feature manually from Isolated Sandbox
    # Logic for ara_count_5d: number of ARA in last 5 trading days before T
    # (Simplified for POC)
    check_df = isolated_df[isolated_df['ticker'] == ticker + ".JK"].sort_values('date').tail(5)
    # ARA check logic (simplified)
    def get_ara_limit(price):
        if price > 5000: return 0.20
        if price > 200: return 0.25
        return 0.35
    
    aras = 0
    # need one more day to calc returns
    hist = isolated_df[isolated_df['ticker'] == ticker + ".JK"].sort_values('date').tail(6)
    for i in range(1, len(hist)):
        prev = hist.iloc[i-1]['close']
        curr = hist.iloc[i]['close']
        ret = (curr - prev) / prev
        limit = get_ara_limit(prev)
        if ret >= (limit - 0.005):
            aras += 1
            
    print(f"Production Value: {prod_val}")
    print(f"Isolated Rebuild: {float(aras)}")
    
    if abs(prod_val - aras) < 1e-5:
        print("VERDICT: PASS (No Lookahead detected in ARA History)")
    else:
        print("VERDICT: FAIL (Lookahead/Logic Mismatch detected)")

if __name__ == "__main__":
    run_isolation_audit("2026-05-06", "ABDA")
