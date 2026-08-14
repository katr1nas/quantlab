import matplotlib
matplotlib.use('agg')

import io
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import telebot


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
from src.data_loader import load_trades
from src.plots import results_distribution, equity_curves

# --- CONFIGURATION ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = telebot.TeleBot(BOT_TOKEN)


def fmt(val, spec=".2f"):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    if isinstance(val, float) and math.isinf(val):
        return "Inf"
    return f"{val:{spec}}"


def run_simulation_and_get_report():
    trades = load_trades()
    if len(trades) == 0:
        raise ValueError("No trades found in dataset.")

    n_simulations = 1000
    n_trades_per_sim = len(trades)

    results = monte_carlo(trades, n_simulations, n_trades_per_sim)
    final_results = results[:, -1]
    drawdowns = np.array([max_drawdown(results[i]) for i in range(n_simulations)])

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

    return report, results, final_results


def fig_to_bytes():
    """Captures the current active Matplotlib plot into RAM."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close('all')
    return buf


# --- COMMAND HANDLERS ---

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Bot ready. Send /run or /mc to execute the Monte Carlo simulation.")


@bot.message_handler(commands=['run', 'mc'])
def handle_run_command(message):
    chat_id = message.chat.id
    bot.send_message(chat_id, "Running Monte Carlo simulation...")

    try:
        report, results, final_results = run_simulation_and_get_report()

        # 1. Send text report
        formatted_message = f"```\n{report}\n```"
        bot.send_message(chat_id, formatted_message, parse_mode="Markdown")

        # 2. Plot & send Equity Curves
        plt.close('all')
        equity_curves(results)
        img1 = fig_to_bytes()
        bot.send_photo(chat_id, img1, caption="Equity Curves")

        # 3. Plot & send Distribution
        plt.close('all')
        results_distribution(final_results)
        img2 = fig_to_bytes()
        bot.send_photo(chat_id, img2, caption="Results Distribution")

    except Exception as e:
        bot.send_message(chat_id, f"Error running simulation: {str(e)}")


if __name__ == "__main__":
    print("Bot is listening for commands...")
    bot.infinity_polling()