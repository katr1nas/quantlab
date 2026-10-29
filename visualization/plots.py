from pathlib import Path


import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

BLUE = "#2563EB"
LIGHT_BLUE = "#DBEAFE"
RED = "#DC2626"
GRID = "#E5E7EB"
MUTED = "#6B7280"


def plot_performance(
    returns: pd.Series,
    title: str = "Strategy Performance",
    initial_capital: float = 1.0,
    save_path: str | Path | None = None,
):
    """Plot compounded equity and percentage drawdown from periodic simple returns."""
    returns = pd.Series(returns).dropna().sort_index()
    equity = initial_capital * (1 + returns).cumprod()
    peak = np.maximum.accumulate(np.r_[initial_capital, equity.to_numpy()])[1:]
    drawdown = pd.Series(equity.to_numpy() / peak - 1, index=equity.index)

    x = returns.index
    is_datetime = isinstance(x, pd.DatetimeIndex)

    fig, (ax_equity, ax_drawdown) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1], "hspace": 0.08},
    )
    fig.patch.set_facecolor("white")

    for ax in (ax_equity, ax_drawdown):
        ax.set_facecolor("white")
        ax.grid(True, color=GRID, linestyle="--", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(length=0, pad=8)
        ax.tick_params(colors=MUTED)

    ax_equity.plot(x, equity, color=BLUE, linewidth=2.0)
    ax_equity.fill_between(x, equity.to_numpy(), initial_capital,
                           color=LIGHT_BLUE, alpha=0.55)
    ax_equity.set_title(title, loc="left", fontsize=16, fontweight="semibold", pad=16)
    ax_equity.set_ylabel("Equity", color=MUTED)
    ax_equity.margins(x=0.01)

    total = equity.iloc[-1] / initial_capital - 1
    max_dd = drawdown.min()
    ax_equity.text(
        0.99, 0.96,
        f"Total return  {total:.2%}     Max drawdown  {max_dd:.2%}",
        transform=ax_equity.transAxes, ha="right", va="top",
        color=MUTED, fontsize=9,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white",
              "edgecolor": GRID, "alpha": 0.95},
    )

    ax_drawdown.fill_between(x, drawdown.to_numpy(), 0, color=RED, alpha=0.16)
    ax_drawdown.plot(x, drawdown, color=RED, linewidth=1.25)
    ax_drawdown.set_ylabel("Drawdown", color=MUTED)
    ax_drawdown.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_drawdown.margins(x=0.01)

    if is_datetime:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax_drawdown.xaxis.set_major_locator(locator)
        ax_drawdown.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    else:
        ax_drawdown.set_xlabel("Observation")

    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.10, hspace=0.12)

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")

    return fig, (ax_equity, ax_drawdown)


def results_distribution(final_results, save_path: str | Path | None = None):
    """Plot the distribution of final Monte Carlo outcomes."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(final_results, bins=50, color=BLUE, alpha=0.85, edgecolor="white")
    ax.axvline(0, color=RED, linestyle="--", linewidth=1.5, label="Break-even")
    ax.set_title("Distribution of Final Outcomes", loc="left")
    ax.set_xlabel("Final return")
    ax.set_ylabel("Frequency")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    return fig, ax



def plot_comparison(strategies: dict):
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, returns in strategies.items():
        equity = (1 + returns).cumprod()
        ax.plot(equity, label=name, linewidth=2)

    ax.set_title("Strategy Comparison", loc="left", fontsize=16)
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity (initial = 1)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=False)

    fig.tight_layout()
    return fig, ax

    
