import numpy as np
import pytest

from src.metrics import (
    expectancy,
    mean,
    median,
    trade_sharpe,
    winrate,
    profit_factor,
    max_drawdown,
    max_consecutive_wins,
    max_consecutive_losses,
)


def test_expectancy_known_values():
    trades = np.array([2.0, 2.0, -1.0, -1.0])  # 50% win, avg win 2, avg loss 1
    assert expectancy(trades) == pytest.approx(0.5 * 2.0 - 0.5 * 1.0)


def test_expectancy_all_wins():
    trades = np.array([1.0, 2.0, 3.0])
    assert expectancy(trades) == pytest.approx(mean(trades))


def test_expectancy_all_losses():
    trades = np.array([-1.0, -2.0])
    assert expectancy(trades) == pytest.approx(mean(trades))


def test_median_odd_length():
    assert median(np.array([3.0, 1.0, 2.0])) == 2.0


def test_median_even_length():
    assert median(np.array([1.0, 2.0, 3.0, 4.0])) == 2.5


def test_mean():
    assert mean(np.array([1.0, 2.0, 3.0])) == 2.0


def test_trade_sharpe_zero_std_returns_zero():
    trades = np.array([1.0, 1.0, 1.0])
    assert trade_sharpe(trades) == 0.0


def test_trade_sharpe_positive_mean_positive_sharpe():
    trades = np.array([2.0, 1.0, 3.0, -1.0])
    assert trade_sharpe(trades) > 0


def test_winrate_basic():
    trades = np.array([1.0, 1.0, -1.0, -1.0])
    assert winrate(trades) == 0.5


def test_winrate_excludes_breakeven():
    trades = np.array([1.0, 0.0, -1.0])
    # total_trades = 3 - 1(be) = 2, wins = 1 -> 0.5
    assert winrate(trades) == 0.5


def test_winrate_all_breakeven_returns_zero():
    trades = np.array([0.0, 0.0])
    assert winrate(trades) == 0


def test_profit_factor_basic():
    trades = np.array([2.0, 2.0, -1.0])
    assert profit_factor(trades) == pytest.approx(4.0)


def test_profit_factor_no_losses_is_inf():
    trades = np.array([1.0, 2.0])
    assert profit_factor(trades) == np.inf


def test_profit_factor_no_wins_no_losses_is_zero():
    trades = np.array([0.0, 0.0])
    assert profit_factor(trades) == 0.0


def test_max_drawdown_monotonic_up_is_zero():
    equity = np.array([1.0, 2.0, 3.0, 4.0])
    assert max_drawdown(equity) == 0.0


def test_max_drawdown_known_dip():
    equity = np.array([10.0, 8.0, 12.0, 5.0, 9.0])
    # running peak hits 12 at index 2, then drops to 5 -> drawdown = -7
    assert max_drawdown(equity) == -7.0


def test_max_consecutive_wins():
    trades = [1, 1, 1, -1, 1, 1]
    assert max_consecutive_wins(trades) == 3


def test_max_consecutive_losses():
    trades = [1, -1, -1, -1, 1, -1]
    assert max_consecutive_losses(trades) == 3


def test_max_consecutive_wins_no_wins():
    trades = [-1, -1, -1]
    assert max_consecutive_wins(trades) == 0