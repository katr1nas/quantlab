import io
import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import telebot

matplotlib.use('agg')

from src.data_loader import load_trades, get_trades_path, append_trade, clear_trades, load_trade_records, filter_trades
from src.metrics import (
    expectancy,
    max_drawdown,
    mean,
    median,
    profit_factor,
    trade_sharpe as sharpe,
    winrate,
)
from src.monte_carlo import monte_carlo
from src.plots import equity_curves, results_distribution

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
STATS_PATH = BASE_DIR / "data" / "stats.json"

load_dotenv(dotenv_path=ENV_PATH)

BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")  # твой личный chat_id, для /stats

bot = telebot.TeleBot(BOT_TOKEN)


def load_stats():
    if not STATS_PATH.exists():
        return {"users": {}, "total_messages": 0}
    try:
        return json.loads(STATS_PATH.read_text())
    except Exception:
        return {"users": {}, "total_messages": 0}


def track_usage(message):
    stats = load_stats()
    chat_id = str(message.chat.id)
    username = message.from_user.username or message.from_user.first_name or "unknown"

    if chat_id not in stats["users"]:
        stats["users"][chat_id] = {"username": username, "messages": 0}

    stats["users"][chat_id]["messages"] += 1
    stats["total_messages"] += 1

    STATS_PATH.parent.mkdir(exist_ok=True)
    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2))


def fmt(val, spec=".2f"):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    if isinstance(val, float) and math.isinf(val):
        return "Inf"
    return f"{val:{spec}}"


def run_simulation_and_get_report(chat_id):
    trades = load_trades(chat_id)
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
        f"Sharpe: {fmt(trade_sharpe(trades))}\n"
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
    track_usage(message)
    bot.reply_to(message, "Bot ready. Send /run or /mc to execute the Monte Carlo simulation.")


@bot.message_handler(commands=['run', 'mc'])
def handle_run_command(message):
    track_usage(message)
    chat_id = message.chat.id
    bot.send_message(chat_id, "Running Monte Carlo simulation...")

    try:
        report, results, final_results = run_simulation_and_get_report(chat_id)

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
    track_usage(message)
    chat_id = message.chat.id
    msg = bot.send_message(
        chat_id,
        "Send: `R ASSET DIRECTION`\nExample: `1.5 EURUSD long` or `-1 XAUUSD short`",
        parse_mode="Markdown"
    )
    bot.register_next_step_handler(msg, process_trade_step)


def parse_trade_line(line):
    parts = line.strip().split()
    if not parts:
        raise ValueError("empty line")
    r = float(parts[0])
    asset = parts[1].upper() if len(parts) > 1 else None
    direction = parts[2].lower() if len(parts) > 2 else None
    if direction and direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got '{direction}'")
    return r, asset, direction


def process_trade_step(message):
    chat_id = message.chat.id
    user_input = message.text.strip()

    try:
        r, asset, direction = parse_trade_line(user_input)
    except ValueError as e:
        bot.send_message(
            chat_id,
            f"Invalid format: {str(e)}\nUse: `R ASSET DIRECTION`, e.g. `1.5 EURUSD long`",
            parse_mode="Markdown"
        )
        return

    try:
        append_trade(chat_id, r, asset, direction)
        bot.send_message(
            chat_id,
            f"Added: `{r:+.2f}R` {asset or ''} {direction or ''}\nRun /run to recalculate metrics.",
            parse_mode="Markdown"
        )
    except Exception as e:
        bot.send_message(chat_id, f"Error saving trade: {str(e)}")


@bot.message_handler(commands=['add_list'])
def handle_add_list(message):
    track_usage(message)
    chat_id = message.chat.id
    msg = bot.send_message(
        chat_id,
        "Send trades, one per line: `R ASSET DIRECTION`\n\n"
        "Example:\n"
        "1.5 EURUSD long\n"
        "-1 XAUUSD short\n"
        "0.5 GBPUSD long",
        parse_mode="Markdown"
    )
    bot.register_next_step_handler(msg, process_list_step)


def process_list_step(message):
    chat_id = message.chat.id
    user_input = message.text.strip()

    if not user_input:
        bot.send_message(chat_id, "The list is empty. Please run /add_list again:)")
        return

    lines = user_input.splitlines()
    parsed = []

    for line_number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            r, asset, direction = parse_trade_line(line)
        except ValueError as e:
            bot.send_message(
                chat_id,
                f"Invalid value on line {line_number}: `{line}` ({e})\n\n"
                "Nothing was added. Please run /add_list again.",
                parse_mode="Markdown"
            )
            return

        parsed.append((r, asset, direction))

    if not parsed:
        bot.send_message(chat_id, "No valid trades found.")
        return

    try:
        for r, asset, direction in parsed:
            append_trade(chat_id, r, asset, direction)

        bot.send_message(
            chat_id,
            f"Successfully added `{len(parsed)}` trades.\nRun /run to recalculate metrics.",
            parse_mode="Markdown"
        )
    except Exception as e:
        bot.send_message(chat_id, f"Error saving trades: {str(e)}")


@bot.message_handler(commands=['clear'])
def handle_clear_command(message):
    track_usage(message)
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
        clear_trades(chat_id)
        bot.send_message(
            chat_id,
            "Successfully cleared all trades. Your trade log was reset to initial state.",
            parse_mode="Markdown"
        )
    except Exception as e:
        bot.send_message(chat_id, f"Error clearing trades: {str(e)}")


@bot.message_handler(commands=['trades'])
def handle_trades(message):
    track_usage(message)
    chat_id = message.chat.id

    try:
        records = load_trade_records(chat_id)
    except ValueError as e:
        bot.send_message(chat_id, str(e))
        return

    lines = [f"Total trades: {len(records)}\n"]
    for i, rec in enumerate(records, start=1):
        r = rec.get("r")
        asset = rec.get("asset") or "-"
        direction = rec.get("direction") or "-"
        lines.append(f"{i}. {r:+.2f}R  {asset}  {direction}")

    text = "\n".join(lines)

    if len(text) > 4000:
        for i in range(0, len(text), 4000):
            bot.send_message(chat_id, text[i:i+4000])
    else:
        bot.send_message(chat_id, text)
def handle_stats(message):
    chat_id = message.chat.id

    if ADMIN_CHAT_ID and str(chat_id) != str(ADMIN_CHAT_ID):
        return  # тихо игнорируем чужих

    stats = load_stats()
    users = stats["users"]

    if not users:
        bot.send_message(chat_id, "Пока никто не пользовался ботом.")
        return

    lines = [f"Всего пользователей: {len(users)}", f"Всего сообщений: {stats['total_messages']}", ""]
    for uid, data in sorted(users.items(), key=lambda x: -x[1]["messages"]):
        lines.append(f"@{data['username']} — {data['messages']} сообщений")

    bot.send_message(chat_id, "\n".join(lines))

@bot.message_handler(commands=['filter'])
def handle_filter(message):
    track_usage(message)
    chat_id = message.chat.id
    msg = bot.send_message(
        chat_id,
        "Send: `DIRECTION, EXCLUDED_ASSET1, EXCLUDED_ASSET2, ...`\n"
        "DIRECTION: `long`, `short`, or leave empty for both.\n"
        "Example: `long, XAUUSD, EURUSD`",
        parse_mode="Markdown"
    )
    bot.register_next_step_handler(msg, process_filter_step)


def process_filter_step(message):
    chat_id = message.chat.id
    parts = [p.strip() for p in message.text.split(",")]

    direction = parts[0].lower() if parts[0] else None
    if direction not in (None, "", "long", "short"):
        bot.send_message(chat_id, f"Invalid direction: `{direction}`", parse_mode="Markdown")
        return
    direction = direction or None
    excluded_assets = [a.upper() for a in parts[1:] if a]

    try:
        trades = filter_trades(chat_id, direction=direction, excluded_assets=excluded_assets)

        n_simulations = 1000
        n_trades_per_sim = len(trades)
        results = monte_carlo(trades, n_simulations, n_trades_per_sim)
        final_results = results[:, -1]
        drawdowns = np.array([max_drawdown(results[i]) for i in range(n_simulations)])

        report = (
            f"Direction: {direction or 'both'}\n"
            f"Excluded: {', '.join(excluded_assets) or 'none'}\n"
            f"Trades: {len(trades)}\n\n"
            "MONTE CARLO SIMULATION\n"
            "---------------------\n"
            f"Mean: {fmt(mean(final_results))}\n"
            f"Median: {fmt(median(final_results))}\n"
            f"Worst DD: {fmt(drawdowns.min())}\n\n"
            "STRATEGY METRICS\n"
            "----------------\n"
            f"Expectancy: {fmt(expectancy(trades), '.4f')}\n"
            f"Profit Factor: {fmt(profit_factor(trades))}\n"
            f"Sharpe: {fmt(sharpe(trades))}\n"
            f"Winrate: {winrate(trades):.2%}"
        )
        bot.send_message(chat_id, f"```\n{report}\n```", parse_mode="Markdown")

        plt.close('all')
        equity_curves(results)
        bot.send_photo(chat_id, fig_to_bytes(), caption="Equity Curves (filtered)")
        plt.close('all')
        results_distribution(final_results)
        bot.send_photo(chat_id, fig_to_bytes(), caption="Distribution (filtered)")
        plt.close('all')

    except ValueError as e:
        bot.send_message(chat_id, str(e))
    except Exception as e:
        bot.send_message(chat_id, f"Error running filtered simulation: {e}")



if __name__ == "__main__":
    bot.set_my_commands([
        telebot.types.BotCommand("start", "Запустить бота"),
        telebot.types.BotCommand("run", "Запустить Monte Carlo симуляцию"),
        telebot.types.BotCommand("add_trade", "Добавить одну сделку"),
        telebot.types.BotCommand("add_list", "Добавить список сделок"),
        telebot.types.BotCommand("clear", "Очистить все сделки"),
        telebot.types.BotCommand("trades", "Показать список сделок"),
        telebot.types.BotCommand("help", "Помощь"),
    ])
    print("Bot is listening for commands...")
    bot.infinity_polling()