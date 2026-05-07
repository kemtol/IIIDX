import pandas as pd
import duckdb
import sys
import numpy as np
from pathlib import Path

def check_parity(target_date: str):
    repo_root = Path(__file__).resolve().parents[2]
    
    # 1. Load Python (L2 Datamart)
    l2_path = repo_root / "data" / "Level_2_Datamart" / "training_datamart_bsjp_overnight.parquet"
    if not l2_path.exists():
        print(f"❌ Python Datamart not found at {l2_path}")
        return

    print(f"🔍 Loading Python features for {target_date}...")
    df_py = pd.read_parquet(l2_path)
    # Filter by date and clean ticker format
    df_py['date'] = pd.to_datetime(df_py['date']).dt.strftime('%Y-%m-%d')
    df_py = df_py[df_py['date'] == target_date].copy()
    df_py['ticker'] = df_py['ticker'].str.replace('.JK', '', regex=False)
    
    if df_py.empty:
        print(f"❌ No Python data found for {target_date} in L2 Datamart.")
        return

    # 2. Load Go (DuckDB Inference Store)
    db_path = repo_root / "inferences" / "bsjp" / "db" / "inference.duckdb"
    if not db_path.exists():
        print(f"❌ DuckDB not found at {db_path}")
        return

    print(f"🔍 Loading Go features from DuckDB for {target_date}...")
    con = duckdb.connect(str(db_path))
    df_go = con.execute(f"SELECT * FROM features_store WHERE date = '{target_date}'").df()
    con.close()

    if df_go.empty:
        print(f"❌ No Go data found in DuckDB for {target_date}. Did you run 'bsjp fetch'?")
        return

    # 3. Align and Compare
    common_tickers = set(df_py['ticker']).intersection(set(df_go['ticker']))
    print(f"📊 Found {len(common_tickers)} common tickers for comparison.")

    df_py = df_py[df_py['ticker'].isin(common_tickers)].sort_values('ticker').set_index('ticker')
    df_go = df_go[df_go['ticker'].isin(common_tickers)].sort_values('ticker').set_index('ticker')

    # Common columns (ignore metadata like date, etc.)
    ignore_cols = ['date', 'fetched_at', 'entry_price', 'exit_price', 'label', 'target_date', 'source', 'status', 'sector', 'industry']
    py_cols = [c for c in df_py.columns if c not in ignore_cols]
    go_cols = [c for c in df_go.columns if c not in ignore_cols]
    
    # Match names
    common_cols = sorted(list(set(py_cols).intersection(set(go_cols))))
    
    # Filter for only numeric columns to avoid string-to-float errors
    numeric_cols = []
    for col in common_cols:
        if pd.api.types.is_numeric_dtype(df_py[col]) and pd.api.types.is_numeric_dtype(df_go[col]):
            numeric_cols = numeric_cols + [col]
    
    print(f"🧪 Comparing {len(numeric_cols)} identical numeric feature names...")

    diffs = []
    for col in numeric_cols:
        # Cast to float32 for fair comparison
        py_vals = df_py[col].astype(np.float32)
        go_vals = df_go[col].astype(np.float32)
        
        # Absolute difference
        delta = (py_vals - go_vals).abs()
        max_delta = delta.max()
        
        if max_delta > 1e-5:
            diffs.append({'feature': col, 'max_delta': max_delta})

    if not diffs:
        print("\n✅ PERFECT PARITY! Python features and Go features are identical.")
    else:
        print(f"\n⚠️ FOUND {len(diffs)} DIVERGENT FEATURES:")
        diff_df = pd.DataFrame(diffs).sort_values('max_delta', ascending=False)
        print(diff_df.head(20).to_string(index=False))
        
    # Check for missing tickers
    py_only = set(df_py.index) - set(df_go.index)
    go_only = set(df_go.index) - set(df_py.index)
    if py_only: print(f"ℹ️ {len(py_only)} tickers only in Python (e.g., {list(py_only)[:5]})")
    if go_only: print(f"ℹ️ {len(go_only)} tickers only in Go (e.g., {list(go_only)[:5]})")

if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else "2026-05-05"
    check_parity(date)
