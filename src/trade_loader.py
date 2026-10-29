from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / 'data/strategies'

def load_return(filename: str, name: str, column: str = 'return') -> pd.Series:
    frame = pd.read_csv(
        OUTPUT_DIR / filename,
        parse_dates=['date'],
        index_col='date',
    )

    returns = frame[column].sort_index()
    returns = returns[~returns.index.duplicated(keep='first')]
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    return returns.rename(name)


