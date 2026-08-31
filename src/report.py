import math

import numpy as np

from src.metrics import expectancy, max_drawdown, mean, median, profit_factor, trade_sharpe, winrate
from src.monte_carlo import monte_carlo


def fmt(val, spec=".2f"):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    if isinstance(val, float) and math.isinf(val):
        return "Inf"
    return f"{val:{spec}}"


def run_monte_carlo_report(trades, n_simulations=1000, meta=None):
    if len(trades) == 0:
        raise ValueError("No trades found in dataset.")

    n_trades_per_sim = len(trades)
    results = monte_carlo(trades, n_simulations, n_trades_per_sim)
    final_results = results[:, -1]
    drawdowns = np.array([max_drawdown(results[i]) for i in range(n_simulations)])

    report = build_report_text(trades, final_results, drawdowns, meta)

    return report, results, final_results


def build_report_text(trades, final_results, drawdowns, meta=None):
    lines = []

    if meta:
        for key, value in meta.items():
            lines.append(f"{key}: {value}")
        lines.append(f"Trades: {len(trades)}")
        lines.append("")

        lines += [
            "MONTE CARLO SIMULATION",
            "---------------------",
            f"Mean: {fmt(mean(final_results))}",
            f"Median: {fmt(median(final_results))}",
            f"Worst DD: {fmt(drawdowns.min())}",
            "",
            "STRATEGY METRICS",
            "----------------",
            f"Expectancy: {fmt(expectancy(trades), '.4f')}",
            f"Profit Factor: {fmt(profit_factor(trades))}",
            f"Sharpe: {fmt(trade_sharpe(trades))}",
            f"Winrate: {winrate(trades):.2%}",
        ]
    else:
        lines += [
            "MONTE CARLO SIMULATION",
            "---------------------",
            f"Mean: {fmt(mean(final_results))}",
            f"Median: {fmt(median(final_results))}",
            f"5%: {fmt(np.percentile(final_results, 5))}",
            f"95%: {fmt(np.percentile(final_results, 95))}",
            "",
            "DRAWDOWN",
            "--------",
            f"Average DD: {fmt(mean(drawdowns))}",
            f"Worst DD: {fmt(drawdowns.min())}",
            f"Loss Prob: {np.mean(final_results < 0):.2%}",
            "",
            "STRATEGY METRICS",
            "----------------",
            f"Expectancy: {fmt(expectancy(trades), '.4f')}",
            f"Mean: {fmt(mean(trades))}",
            f"Median: {fmt(median(trades))}",
            f"Profit Factor: {fmt(profit_factor(trades))}",
            f"Sharpe: {fmt(trade_sharpe(trades))}",
            f"Winrate: {winrate(trades):.2%}",
        ]

    return "\n".join(lines)