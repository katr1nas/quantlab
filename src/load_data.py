import yfinance as yf
import pandas as pd


def load_data_yf(symbol: str, start: str, end: str) -> pd.Series:
    df = yf.download(symbol, start=start, end=end)
    df.columns = df.columns.droplevel(1)
    df.columns = [c.lower() for c in df.columns]
    return df
