import io
import math
import os
from pathlib import Path

from dotenv import load_dotenv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import telebot

matplotlib.use('agg')

from src.data_loader import load_trades
from src.metrics import (
    expectancy,
    max_drawdown,
    mean,
    median,
    profit_factor,
    sharpe,
    winrate,
)
from src.monte_carlo import monte_carlo
from src.plots import equity_curves, results_distribution

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

bot = telebot.TeleBot(BOT_TOKEN)
TRADES_FILE = "data/trades.csv"


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


@bot.message_handler(commands=['add_trade'])
def handle_add_trade(message):
    chat_id = message.chat.id
    msg = bot.send_message(
        chat_id,    
        'Please enter trade result in R: (e.g., `1.5` or `-1.0`)',
        parse_mode="Markdown" 
    )
    bot.register_next_step_handler(msg, process_trade_step)


def process_trade_step(message):
    chat_id = message.chat.id
    user_input = message.text.strip()

    try:
        trade_r = float(user_input)
    except ValueError:
        bot.send_message(
            chat_id, 
            "Invalid format. Please enter a numeric value (e.g., 1.5 or -1.0)."
        )
        return

    try:
        os.makedirs(os.path.dirname(TRADES_FILE), exist_ok=True)
        
        file_exists = os.path.exists(TRADES_FILE) and os.path.getsize(TRADES_FILE) > 0
        with open(TRADES_FILE, "a", encoding="utf-8") as f:
            if file_exists:
                f.write("\n")
            f.write(f"{trade_r}")

        bot.send_message(
            chat_id, 
            f"Successfully added trade: `{trade_r:+.2f}R`\nRun /run to recalculate metrics.",
            parse_mode="Markdown"
        )
    except Exception as e:
        bot.send_message(chat_id, f"Error saving trade: {str(e)}")


@bot.message_handler(commands=['add_list'])
def handle_add_list(message):  # Fixed handler function name
    chat_id = message.chat.id
    msg = bot.send_message(
        chat_id,    
        "Send trades in R, one per line.\n\n"
        "Example:\n"
        "1.5\n"
        "-1\n"
        "0.5\n"
        "2\n"
        "-0.8",
        parse_mode="Markdown" 
    )
    bot.register_next_step_handler(msg, process_list_step)


def process_list_step(message):
    chat_id = message.chat.id
    user_input = message.text.strip()

    if not user_input:
        bot.send_message(
            chat_id, 
            "The list is empty. Please run /add_list again:)"
        )
        return
    
    lines = user_input.splitlines()
    trades_r = []

    for line_number, line in enumerate(lines, start=1):
        line = line.strip()

        if not line:
            continue
        
        try:
            trade_r = float(line)
        except ValueError:
            bot.send_message(
                chat_id,
                f"Invalid value on line {line_number}: `{line}`\n\n"
                "Nothing was added. Please run /add_list again.",
                parse_mode="Markdown"
            )
            return
        
        trades_r.append(trade_r)
    
    if not trades_r:
        bot.send_message(
            chat_id,
            "No valid trades found."
        )
        return

    try:
        os.makedirs(os.path.dirname(TRADES_FILE), exist_ok=True)
        
        file_exists = os.path.exists(TRADES_FILE) and os.path.getsize(TRADES_FILE) > 0
        with open(TRADES_FILE, "a", encoding="utf-8") as f:
            if file_exists:
                f.write("\n")
            f.write("\n".join(str(r) for r in trades_r))

        bot.send_message(
            chat_id,
            f"Successfully added `{len(trades_r)}` trades.\n"
            "Run /run to recalculate metrics.",
            parse_mode="Markdown"
        )
    except Exception as e:
        bot.send_message(
            chat_id,
            f"Error saving trades: {str(e)}"
        )
@bot.message_handler(commands=['clear'])
def handle_clear_command(message):
    chat_id = message.chat.id
    msg = bot.send_message(
        chat_id,
        "⚠️ *Are you sure you want to clear all trades?*\nType `YES` to confirm.",
        parse_mode="Markdown"
    )
    bot.register_next_step_handler(msg, process_clear_confirmation)


def process_clear_confirmation(message):
    chat_id = message.chat.id
    user_input = message.text.strip().upper()

    if user_input != "YES":
        bot.send_message(chat_id, "Action canceled. Trades were not cleared.")
        return

    try:
        # Overwrite file keeping only the header 'R'
        os.makedirs(os.path.dirname(TRADES_FILE), exist_ok=True)
        with open(TRADES_FILE, "w") as f:
            f.write("R\n")

        bot.send_message(
            chat_id, 
            "Successfully cleared all trades. `trades.csv` reset to initial state.",
            parse_mode="Markdown"
        )
    except Exception as e:
        bot.send_message(chat_id, f"Error clearing trades: {str(e)}")


if __name__ == "__main__":
    print("Bot is listening for commands...")
    bot.infinity_polling()