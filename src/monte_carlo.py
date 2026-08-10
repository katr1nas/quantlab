#%%
import numpy as np
import pandas as pd
from pathlib import Path
import importlib
import data_loader
import plots
import metrics

importlib.reload(data_loader)

from data_loader import load_trades
from plots import results_distribution
from metrics import expectancy
from data_loader import bootstrap
from metrics import winrate
from metrics import profit_factor
from metrics import max_drawdown
from metrics import sharpe
from metrics import mean
from metrics import median



#%%
trades = load_trades()

if len(trades) == 0:
    raise ValueError("Put at least one trade.")

#%%
n_simulations = 1000
n_trades_per_sim = len(trades)

#%%
results = np.zeros((n_simulations, n_trades_per_sim))
for i in range(n_simulations):
    sample = np.random.choice(trades, size=n_trades_per_sim, replace=True)
    results[i] = np.cumsum(sample)

final_results = results[:, -1]

drawdowns = np.array([max_drawdown(results[i]) for i in range(n_simulations)])
# %%

print("Mean:",  mean(final_results))
print("Median:", median(final_results))
print("5%:", np.percentile(final_results, 5))
print("95%:", np.percentile(final_results, 95))
# %%
print("Average drawdown:", mean(drawdowns))
print("Worst drawdown:", drawdowns.min())
# %%
prob_loss = np.mean(final_results < 0)
print(f"Probability of loss: {prob_loss:.2f}")
# %%
results_distribution(final_results)
# %%
print(f"Expecatncy: {expectancy(trades)}")

# %%
print(f"Bootstrap statisctics {bootstrap(trades)}")
# %%
print(f"Mean: {mean(trades)}")
#%%
print(f"Median: {median(trades)}")

#%%
print(f"Profit factor: {profit_factor(trades)}")
#%%
print(f"Sharpe: {sharpe(trades)}")
#%%
print(f"Winrate: {winrate(trades)}")

# %%
