#%%
import numpy as np

#%%
def expectancy(trades):
    wins = trades[trades > 0]
    losses = trades[trades < 0]
    
    win_rate = len(wins) / len(trades) if len(trades) > 0 else 0
    loss_rate = len(losses) / len(trades) if len(trades) > 0 else 0
    
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = np.abs(losses.mean()) if len(losses) > 0 else 0.0
    
    return (win_rate * avg_win) - (loss_rate * avg_loss)

#%%
def median(trades):
    trades = np.sort(trades)
    m = len(trades)
    if m % 2 == 0: # if m is even then median is average between 2 items in center
        median = (trades[m // 2 - 1] + trades[m // 2]) / 2
    else: 
        median = trades[m // 2]
    return median


#%%
def mean(trades):
    return np.mean(trades)

#%%
def trade_sharpe(trades):
    mean_trades = mean(trades)
    std_trades = np.std(trades, ddof=1)
    if std_trades == 0:
        return 0.0
    
    return mean_trades / std_trades
#%%
def winrate(trades):
    wins = trades[trades > 0]
    be = trades[trades == 0]
    total_trades = len(trades) - len(be)

    if total_trades == 0:
        return 0

    return len(wins) / total_trades



#%%
def profit_factor(trades):
    wins = trades[trades > 0]
    losses = trades[trades < 0]
    
    if len(losses) == 0:
        return np.inf if len(wins) > 0 else 0.0
    
    return np.sum(wins) / np.abs(np.sum(losses))
    

#%%
def max_drawdown(equity_curve):
    peak = np.maximum.accumulate(equity_curve)
    return (equity_curve - peak).min()

    