#%%
import matplotlib.pyplot as plt

#%%
def results_distribution(final_results):
    plt.hist(final_results, bins=50)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Final R")
    plt.ylabel("Frequency")
    plt.show()