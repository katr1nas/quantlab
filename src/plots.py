#%%
import matplotlib
matplotlib.use('agg')
#%%
import matplotlib.pyplot as plt


#%%
def results_distribution(final_results):
    plt.hist(final_results, bins=50)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Final R")
    plt.ylabel("Frequency")

#%%
def equity_curves(results, n_curves=100):
    plt.figure(figsize=(10, 5))
    for i in range(n_curves):
        plt.plot(results[i])
    
    plt.title("Monte Carlo Equity Curves")
    plt.xlabel("Trade")
    plt.ylabel("Cumulative R")
    
    plt.grid(True)
# %%
