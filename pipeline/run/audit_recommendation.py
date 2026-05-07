import pandas as pd
import duckdb
import lightgbm as lgb
import sys
import numpy as np
from pathlib import Path

def compare_recommendations(target_date: str):
    repo_root = Path(__file__).resolve().parents[2]
    model_path = repo_root / "model" / "BSJP" / "bsjp_v19d_close10_preclose14_orb_md100_l21.5" / "model_lightgbm_opening_tp3.txt"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return

    # 1. Load Model
    bst = lgb.Booster(model_file=str(model_path))
    model_features = bst.feature_name()

    # 2. Load Python Data (L2 + Modules)
    l2_path = repo_root / "data" / "Level_2_Datamart" / "training_datamart_bsjp_overnight.parquet"
    p14_path = repo_root / "data" / "Level_1_Features" / "modules" / "preclose14_features.parquet"
    
    df_py = pd.read_parquet(l2_path)
    df_py['date_str'] = pd.to_datetime(df_py['date']).dt.strftime('%Y-%m-%d')
    df_py = df_py[df_py['date_str'] == target_date].copy()
    # Normalize ticker (remove .JK if present)
    df_py['ticker'] = df_py['ticker'].str.replace('.JK', '', regex=False)
    
    if p14_path.exists():
        df_p14 = pd.read_parquet(p14_path)
        df_p14['date_str'] = pd.to_datetime(df_p14['date']).dt.strftime('%Y-%m-%d')
        df_p14 = df_p14[df_p14['date_str'] == target_date].copy()
        df_p14['ticker'] = df_p14['ticker'].str.replace('.JK', '', regex=False)
        
        # Join
        df_py = df_py.merge(df_p14, on=['date_str', 'ticker'], how='left', suffixes=('', '_mod'))
    
    if df_py.empty:
        print(f"❌ No Python data for {target_date}")
        return
    
    # 3. Predict using Python
    # Fill missing features with 0, select only model features
    X = df_py.reindex(columns=model_features).fillna(0).astype(np.float32)
    df_py['proba'] = bst.predict(X)
    top3_py = df_py.sort_values('proba', ascending=False).head(3)

    # 4. Load Go Data (DuckDB)
    db_path = repo_root / "inferences" / "bsjp" / "db" / "inference.duckdb"
    con = duckdb.connect(str(db_path))
    # Note: Go might have tickers with .JK or not, but DuckDB usually has clean names.
    # Our parity audit showed Go tickers match Python tickers.
    df_go = con.execute(f"SELECT * FROM features_store WHERE date = '{target_date}'").df()
    con.close()
    
    if df_go.empty:
        print(f"⚠️ Go data for {target_date} not found. Predicting now...")
        # I won't run bash here, I'll just warn.
        return

    # 5. Predict using Python on Go's data to see if Go's features produce same rank
    X_go = df_go[model_features].fillna(0).astype(np.float32)
    df_go['proba'] = bst.predict(X_go)
    top3_go = df_go.sort_values('proba', ascending=False).head(3)

    print(f"\n=== Recommendation Parity for {target_date} ===")
    print("\n[PYTHON VERSION (L2 Parquet)]")
    print(top3_py[['ticker', 'proba']].to_string(index=False))
    
    print("\n[GO VERSION (DuckDB Features)]")
    print(top3_go[['ticker', 'proba']].to_string(index=False))

    py_tickers = top3_py['ticker'].tolist()
    go_tickers = top3_go['ticker'].tolist()
    
    if py_tickers == go_tickers:
        print("\n✅ RANK PARITY MATCHED! Both versions recommend the same Top 3 tickers.")
    else:
        print("\n❌ RANK DIVERGENCE! Different Top 3 detected.")

if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else "2026-03-30"
    compare_recommendations(date)
