#%%
import pandas as pd
import numpy as np
from pathlib import Path
from pandas.errors import EmptyDataError

#%%
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

#%%
def get_trades_path(chat_id):
    return DATA_DIR / f"trades_{chat_id}.csv"

#%%
def load_trades(chat_id):
    path = get_trades_path(chat_id)

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise ValueError("No trades yet. Use /add_trade or /add_list first.")
    except EmptyDataError:
        raise ValueError("trades.csv is empty. Add at least one trade.")

    if "R" not in df.columns:
        raise ValueError("Column 'R' is missing.")

    return df["R"].to_numpy(dtype=float)

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