import numpy as np
import pandas as pd


def expectancy(trades):
    wins = trades[trades > 0]
    losses = trades[trades < 0]

    win_rate = len(wins) / len(trades) if len(trades) > 0 else 0
    loss_rate = len(losses) / len(trades) if len(trades) > 0 else 0

    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = np.abs(losses.mean()) if len(losses) > 0 else 0.0

    return (win_rate * avg_win) - (loss_rate * avg_loss)


def median(trades):
    trades = np.sort(trades)
    m = len(trades)
    if m % 2 == 0:
        median = (trades[m // 2 - 1] + trades[m // 2]) / 2
    else:
        median = trades[m // 2]
    return median


def mean(trades):
    return np.mean(trades)


def sharpe(returns, periods_per_year=252, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate / periods_per_year

    return (
        np.mean(excess_returns)
        / np.std(excess_returns, ddof=1)
        * np.sqrt(periods_per_year)
    )

def sortino(returns, T=0.0, periods_per_year=252):
    r = np.asarray(returns, dtype=float)
    downside = np.minimum(r - T, 0.0)
    downside_dev = np.sqrt(np.mean(downside ** 2))
    if downside_dev == 0:
        return np.nan
    return (r.mean() - T) / downside_dev * np.sqrt(periods_per_year)

def cagr(returns, periods_per_year=252):
    returns_prod = np.prod(1 + returns)

    return returns_prod ** (periods_per_year / len(returns)) - 1

def annualized_volatility(returns, period_per_year=252):
    return np.std(returns, ddof=1) * np.sqrt(period_per_year)
    
def calmar(returns, periods_per_year=252):
    equity = (1 + returns).cumprod()
    return cagr(returns) / abs(max_drawdown(equity))

def winrate(trades):
    wins = trades[trades > 0]
    be = trades[trades == 0]
    total_trades = len(trades) - len(be)

    if total_trades == 0:
        return 0

    return len(wins) / total_trades


def profit_factor(trades):
    wins = trades[trades > 0]
    losses = trades[trades < 0]

    if len(losses) == 0:
        return np.inf if len(wins) > 0 else 0.0

    return np.sum(wins) / np.abs(np.sum(losses))


def max_drawdown(equity_curve):
    equity_curve = np.r_[1, equity_curve]
    peak = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve / peak - 1
    return drawdown.min()

def max_consecutive_losses(trades):
    max_streak = 0
    current = 0
    for x in trades:
        if x < 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def max_consecutive_wins(trades):
    max_streak = 0
    current = 0
    for x in trades:
        if x > 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()

def total_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).prod() - 1

def recovery_factor(returns):
    equity = equity_curve(returns)

    return total_returns(returns) / abs(max_drawdown(equity))

def calculate_metrics(returns, periods_per_year=252):
    equity = equity_curve(returns)

    return {
        "total_return": total_returns(returns),
        "cagr": cagr(returns, periods_per_year),
        "annualized_volatility": annualized_volatility(
            returns, periods_per_year
        ),
        "sharpe": sharpe(returns, periods_per_year),
        "sortino": sortino(returns, periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar(returns, periods_per_year),
        "recovery_factor": recovery_factor(returns),
        "max_consecutive_wins": max_consecutive_wins(returns),
        'max_consecutive_losses': max_consecutive_losses(returns),
    }

