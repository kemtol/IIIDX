import pytest
from datetime import datetime, timedelta
import pandas as pd

import sys
from pathlib import Path

# Add scripts directory to path to import the script
scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
sys.path.append(str(scripts_dir))

from rolling_validation_v23 import generate_windows

def test_generate_windows_basic():
    # Create a mock list of consecutive dates
    base_date = datetime(2026, 1, 1)
    trading_days = [base_date + timedelta(days=i) for i in range(200)]
    
    windows = generate_windows(trading_days, num_windows=3, step_days=20, oot_days=50)
    
    assert len(windows) == 3
    
    # Window 0 (most recent)
    w0 = windows[0]
    assert w0["window_idx"] == 0
    assert w0["train_end_date"] == trading_days[-51]
    assert w0["oot_start_date"] == trading_days[-50]
    assert w0["oot_end_date"] == trading_days[-1]
    assert len(w0["oot_dates"]) == 50
    
    # Window 1 (shifted by 20 days)
    w1 = windows[1]
    assert w1["window_idx"] == 1
    assert w1["train_end_date"] == trading_days[-71]
    assert w1["oot_start_date"] == trading_days[-70]
    assert w1["oot_end_date"] == trading_days[-21]
    assert len(w1["oot_dates"]) == 50
    
    # Window 2 (shifted by 40 days)
    w2 = windows[2]
    assert w2["window_idx"] == 2
    assert w2["train_end_date"] == trading_days[-91]
    assert w2["oot_start_date"] == trading_days[-90]
    assert w2["oot_end_date"] == trading_days[-41]
    assert len(w2["oot_dates"]) == 50

def test_generate_windows_insufficient_days():
    # Only 50 days available, but we need OOT (50) + some train days + shifts
    base_date = datetime(2026, 1, 1)
    trading_days = [base_date + timedelta(days=i) for i in range(50)]
    
    with pytest.raises(ValueError, match="Insufficient trading days"):
        generate_windows(trading_days, num_windows=2, step_days=20, oot_days=50)
