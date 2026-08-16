#%%
import pandas as pd
import numpy as np
from pathlib import Path
from pandas.errors import EmptyDataError
import json

#%%
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

#%%
def get_trades_path(chat_id):
    return DATA_DIR / f"trades_{chat_id}.json1"

#%%
def load_trades_records(chat_id):
    path = get_trades_path(chat_id)

    if not path.exists():
        raise ValueError("No trades yet. Use /add_trade or /add_list first.")

    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    
    if not records:
        raise ValueError('No trades yet. Use /add_trade or /add_list first.')
    
    return records
#%%
def load_trades(chat_id, asset=None, direction=None):
    records = load_trades_records(chat_id)

    if asset is not None:
        records = [r for r in records if r.get("asset", "").upper() == asset.upper()]
    if direction is not None:
        records = [r for r in records if r.get("direction", "").upper() == direction.upper()]
    
    if not records:
        raise ValueError("No trades match the given filters.")
    
    return np.array([r["r"] for r in records], dtype=float)

def append_trade(chat_id, r, asset=None, direction=None):
    path = get_trades_path(chat_id)
    path.parent.mkdir(exist_ok=True)

    record = {"r": r, "asset": asset, "direction": direction}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def clear_trades(chat_id):
    path = get_trades_path(chat_id)
    path.parent.mkdir(exist_ok=True)
    path.write_text("")



#%%
def bootstrap(trades, bootstrap_n=10000):
    n = len(trades)
    rng = np.random.default_rng(seed=42)

    boot_samples = rng.choice(trades, size=(bootstrap_n, n), replace=True)

    boot_stats = np.median(boot_samples, axis=1)

    point_estimate = np.median(trades)
    std_error = np.std(boot_stats, ddof=1)
    ci_lower, ci_upper = np.percentile(boot_stats, [2.5, 97.5])

    return point_estimate, std_error, ci_lower, ci_upper


# %%