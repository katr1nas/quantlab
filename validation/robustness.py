import numpy as np


def clean_returns(returns):
    returns = returns.drop_duplicates(subset=['data'])


    returns.replace[[np.inf, -np.inf], np.nan]
    returns.dropna()
    