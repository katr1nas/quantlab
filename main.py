import io
import math
import requests
import numpy as np
import matplotlib.pyplot as plt

# Replace imports from your modules if needed; using dummy functions here for isolated testing
from src.monte_carlo import monte_carlo
from src.metrics import (
    expectancy,
    mean,
    median,
    sharpe,
    profit_factor,
    max_drawdown,
    winrate,
)
from src.plots import results_distribution, equity_curves


def fmt(val, spec=".2f"):
    """Safely format values for string output."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    if isinstance(val, float) and math.isinf(val):
        return "Inf"
    return f"{val:{spec}}"


def send_telegram_message(text: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    response = requests.post(url, json=payload)
    if not response.ok:
        print(f"[Telegram Error - Message]: {response.status_code} -> {response.text}")
    else:
        print("[Telegram]: Message delivered successfully.")


def send_telegram_photo(fig, caption: str = ""):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    
    files = {'photo': ('plot.png', buf, 'image/png')}
    data = {'chat_id': CHAT_ID, 'caption': caption}
    response = requests.post(url, data=data, files=files)
    if not response.ok:
        print(f"[Telegram Error - Photo]: {response.status_code} -> {response.text}")
    else:
        print(f"[Telegram]: Photo ({caption}) delivered successfully.")
    plt.close(fig)


def get_mock_trades():
    """
    Returns a sample numpy array of PnL trade returns containing
    both positive and negative values to prevent divide-by-zero errors.
    """
    return np.array([
        150.0, -80.0, 210.0, -110.0, 95.0, 
        -45.0, 310.0, -120.0, 180.0, -90.0, 
        220.0, -130.0, 105.0, -60.0, 140.0
    ])


def main():
    # Use mock trade list containing mixed profits and losses
    trades = get_mock_trades()

    if len(trades) == 0:
        raise ValueError("Put at least one trade.")

    n_simulations = 1000
    n_trades_per_sim = len(trades)

    results = monte_carlo(
        trades,
        n_simulations,
        n_trades_per_sim
    )

    final_results = results[:, -1]

    drawdowns = np.array([
        max_drawdown(results[i])
        for i in range(n_simulations)
    ])

    report = (
        "MONTE CARLO SIMULATION\n"
        "---------------------\n"
        f"Mean: {fmt(mean(final_results))}\n"
        f"Median: {fmt(median(final_results))}\n"
        f"5%: {fmt(np.percentile(final_results, 5))}\n"
        f"95%: {fmt(np.percentile(final_results, 95))}\n\n"
        "DRAWDOWN\n"
        "--------\n"
        f"Average DD: {fmt(mean(drawdowns))}\n"
        f"Worst DD: {fmt(drawdowns.min())}\n"
        f"Loss Prob: {np.mean(final_results < 0):.2%}\n\n"
        "STRATEGY METRICS\n"
        "----------------\n"
        f"Expectancy: {fmt(expectancy(trades), '.4f')}\n"
        f"Mean: {fmt(mean(trades))}\n"
        f"Median: {fmt(median(trades))}\n"
        f"Profit Factor: {fmt(profit_factor(trades))}\n"
        f"Sharpe: {fmt(sharpe(trades))}\n"
        f"Winrate: {winrate(trades):.2%}"
    )

    formatted_message = f"```\n{report}\n```"

    # Print to console
    print(report)
    print("\nSending to Telegram...\n")

    # Send text block
    send_telegram_message(formatted_message)

    # Send plots
    plt.close('all')

    # 1. Generate and capture Equity Curves
    equity_curves(results)
    fig1 = plt.gcf()  # Grab the figure that equity_curves() just drew
    send_telegram_photo(fig1, caption="Equity Curves")

    plt.close('all')

    # 2. Generate and capture Distribution
    results_distribution(final_results)
    fig2 = plt.gcf()  # Grab the figure that results_distribution() just drew
    send_telegram_photo(fig2, caption="Results Distribution")
    
    plt.close('all')


if __name__ == "__main__":
    main()