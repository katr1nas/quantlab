#%%
import numpy as np

#%%
def monte_carlo(trades, n_simulations, n_trades_per_sim, seed=42):
    results = np.zeros((n_simulations, n_trades_per_sim))
    rng = np.random.default_rng(seed=seed)
    for i in range(n_simulations):
        sample = rng.choice(trades, size=n_trades_per_sim, replace=True)
        results[i] = np.cumsum(sample)
    
    return results
