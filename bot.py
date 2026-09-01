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

from src.data_loader import (
    load_trades,
    get_trades_path,
    append_trade,
    clear_trades,
    load_trade_records,
    filter_trades,
    DATA_DIR,
)
from src.metrics import (
    expectancy,
    max_drawdown,
    mean,
    median,
    profit_factor,
    trade_sharpe,
    winrate,
)
from src.monte_carlo import monte_carlo
from src.plots import equity_curves, results_distribution
from src.mt5_parser import parse_mt5_html
from src.report import run_monte_carlo_report
from src.ml.win_predictor import (
    train_win_model,
    save_model,
    load_model,
    predict_win_probability,
    predict_batch,
    evaluate_on_holdout,
)

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
STATS_PATH = BASE_DIR / "data" / "stats.json"

load_dotenv(dotenv_path=ENV_PATH)

BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")

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
    return run_monte_carlo_report(trades)


def fig_to_bytes():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close('all')
    return buf

# commad handlers

@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    track_usage(message)

    bot.reply_to(
        message,
        "Welcome to QuantLab \n\n"
        "Analyze your trading journal in R-multiples: find your edge, "
        "understand possible drawdowns, and explore what the next 50 or "
        "100 trades could look like.\n\n"
        "Start by adding your trades:\n"
        "/add_trade — add one trade\n"
        "/add_list — paste several trades\n\n"
        "When your trades are added, use /run to generate your report.\n\n"
        "New to R? 1R is the amount you risked on one trade; -1R is a full planned loss."
    )


@bot.message_handler(commands=['instructions'])
def handle_instructions(message):
    track_usage(message)
    chat_id = message.chat.id

    text = (
        "WHAT THE ML PREDICTIONS MEAN\n"
        "-----------------------------\n"
        "/predict, /predict_all and /backtest_model do NOT forecast future "
        "price. They estimate the probability that a trade with given "
        "traits (asset, direction, session, hour) would have been a WIN, "
        "based on the statistical pattern in your past trades.\n\n"
        "How it works:\n"
        "1. /train_model fits a gradient boosting model on your stored "
        "trades: input = asset/direction/session/hour/day_of_week, "
        "output = WIN (r > 0) or LOSS.\n"
        "2. /predict takes a hypothetical trade and returns a probability, "
        "e.g. 67% — the model's estimate based on your trading history, "
        "nothing more.\n"
        "3. /backtest_model checks the model on trades it never trained on "
        "(holdout). This is the number that actually reflects skill.\n\n"
        "How to read the numbers:\n"
        "- Accuracy ~50% = the model guesses no better than a coin flip.\n"
        "- AUC ~0.5 = same thing, measured differently (0.5 = random, "
        "1.0 = perfect separation of WIN/LOSS).\n"
        "- AUC meaningfully above 0.5 (e.g. 0.65+) on the HOLDOUT set is "
        "the only signal worth trusting. High training accuracy with low "
        "holdout accuracy just means the model memorized noise.\n\n"
        "A prediction is a statistical estimate from historical data, "
        "never a guarantee. Trade counts under a few hundred are usually "
        "too small for the model to find anything real — expect AUC near "
        "0.5 until you have real, sizeable history."
    )
    bot.send_message(chat_id, text)


@bot.message_handler(commands=['run'])
def handle_run_command(message):
    track_usage(message)
    chat_id = message.chat.id
    bot.send_message(chat_id, "Running Monte Carlo simulation...")

    try:
        report, results, final_results = run_simulation_and_get_report(chat_id)

        formatted_message = f"```\n{report}\n```"
        bot.send_message(chat_id, formatted_message, parse_mode="Markdown")

        plt.close('all')
        equity_curves(results)
        img1 = fig_to_bytes()
        bot.send_photo(chat_id, img1, caption="Equity Curves")

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
        "Send: `R ASSET DIRECTION [HH:MM]`\n"
        "Time is optional (UTC, 24h) — defaults to now if omitted.\n"
        "Example: `1.5 EURUSD long` or `-1 XAUUSD short 14:30`",
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
    time_str = parts[3] if len(parts) > 3 else None
    if direction and direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got '{direction}'")
    if time_str:
        try:
            hh, mm = time_str.split(":")
            if not (0 <= int(hh) <= 23 and 0 <= int(mm) <= 59):
                raise ValueError
        except ValueError:
            raise ValueError(f"invalid time '{time_str}', expected HH:MM (24h)")
    return r, asset, direction, time_str


def process_trade_step(message):
    chat_id = message.chat.id
    user_input = message.text.strip()

    try:
        r, asset, direction, time_str = parse_trade_line(user_input)
    except ValueError as e:
        bot.send_message(
            chat_id,
            f"Invalid format: {str(e)}\nUse: `R ASSET DIRECTION [HH:MM]`, e.g. `1.5 EURUSD long 14:30`",
            parse_mode="Markdown"
        )
        return

    try:
        append_trade(chat_id, r, asset, direction, time_str)
        bot.send_message(
            chat_id,
            f"Added: `{r:+.2f}R` {asset or ''} {direction or ''} {time_str or ''}\nRun /run to recalculate metrics.",
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
        "Send trades, one per line: `R ASSET DIRECTION [HH:MM]`\n"
        "Time is optional (UTC, 24h) — defaults to now if omitted.\n\n"
        "Example:\n"
        "1.5 EURUSD long 09:15\n"
        "-1 XAUUSD short 21:40\n"
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
            r, asset, direction, time_str = parse_trade_line(line)
        except ValueError as e:
            bot.send_message(
                chat_id,
                f"Invalid value on line {line_number}: `{line}` ({e})\n\n"
                "Nothing was added. Please run /add_list again.",
                parse_mode="Markdown"
            )
            return

        parsed.append((r, asset, direction, time_str))

    if not parsed:
        bot.send_message(chat_id, "No valid trades found.")
        return

    try:
        for r, asset, direction, time_str in parsed:
            append_trade(chat_id, r, asset, direction, time_str)

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


@bot.message_handler(commands=['stats'])
def handle_stats(message):
    chat_id = message.chat.id

    if ADMIN_CHAT_ID and str(chat_id) != str(ADMIN_CHAT_ID):
        return

    stats = load_stats()
    users = stats["users"]

    if not users:
        bot.send_message(chat_id, "No usage yet.")
        return

    lines = [f"Users: {len(users)}", f"Total messages: {stats['total_messages']}", ""]
    for uid, data in sorted(users.items(), key=lambda x: -x[1]["messages"]):
        lines.append(f"@{data['username']} — {data['messages']} messages")

    bot.send_message(chat_id, "\n".join(lines))


@bot.message_handler(commands=['filter'])
def handle_filter(message):
    track_usage(message)
    chat_id = message.chat.id
    msg = bot.send_message(
        chat_id,
        "Send: `DIRECTION, SESSION, EXCLUDED_ASSET1, EXCLUDED_ASSET2, ...`\n"
        "DIRECTION: `long`, `short`, or leave empty for both.\n"
        "SESSION: `Tokyo`, `Frankfurt`, `London`, `NewYork`, `Overlap`, or leave empty for all.\n"
        "Example: `long, London, XAUUSD, EURUSD`",
        parse_mode="Markdown"
    )
    bot.register_next_step_handler(msg, process_filter_step)


def process_filter_step(message):
    chat_id = message.chat.id
    parts = [p.strip() for p in message.text.split(",")]

    direction = parts[0].lower() if parts and parts[0] else None
    if direction not in (None, "", "long", "short"):
        bot.send_message(chat_id, f"Invalid direction: `{direction}`", parse_mode="Markdown")
        return
    direction = direction or None

    session = parts[1] if len(parts) > 1 and parts[1] else None
    excluded_assets = [a.upper() for a in parts[2:] if a]

    try:
        trades = filter_trades(chat_id, direction=direction, excluded_assets=excluded_assets, session=session)

        meta = {
            "Direction": direction or "both",
            "Session": session or "all",
            "Excluded": ", ".join(excluded_assets) or "none",
        }
        report, results, final_results = run_monte_carlo_report(trades, meta=meta)
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


@bot.message_handler(commands=['import_mt5'])
def handle_import_mt5(message):
    track_usage(message)
    chat_id = message.chat.id
    msg = bot.send_message(
        chat_id,
        "Send account balance and risk % per trade, e.g. `50000 1` (=1% risk).\n"
        "Then I'll ask you to upload the MT5 History HTML report.",
        parse_mode="Markdown"
    )
    bot.register_next_step_handler(msg, process_mt5_balance_step)


def process_mt5_balance_step(message):
    chat_id = message.chat.id
    try:
        balance_str, risk_str = message.text.strip().split()
        balance = float(balance_str)
        risk_pct = float(risk_str) / 100.0
        if balance <= 0 or not (0 < risk_pct < 1):
            raise ValueError
    except ValueError:
        bot.send_message(chat_id, "Invalid format. Use: `50000 1`", parse_mode="Markdown")
        return

    msg = bot.send_message(chat_id, "Now upload the MT5 History report (.html file).")
    bot.register_next_step_handler(msg, process_mt5_file_step, balance, risk_pct)


def process_mt5_file_step(message, balance, risk_pct):
    chat_id = message.chat.id

    if not message.document:
        bot.send_message(chat_id, "That's not a file. Please upload the .html report, or run /import_mt5 again.")
        return

    file_info = bot.get_file(message.document.file_id)
    downloaded = bot.download_file(file_info.file_path)

    tmp_path = DATA_DIR / f"mt5_import_{chat_id}.html"
    DATA_DIR.mkdir(exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(downloaded)

    try:
        trades = parse_mt5_html(str(tmp_path), balance, risk_pct)
    except Exception as e:
        bot.send_message(chat_id, f"Failed to parse report: {e}")
        return
    finally:
        tmp_path.unlink(missing_ok=True)

    for t in trades:
        append_trade(chat_id, t["r"], t["asset"], t["direction"])

    bot.send_message(
        chat_id,
        f"Imported {len(trades)} trades from MT5 (balance={balance:.0f}, risk={risk_pct*100:.2f}%).\n"
        "Run /run to recalculate metrics.",
        parse_mode="Markdown"
    )


@bot.message_handler(commands=['train_model'])
def handle_train_model(message):
    track_usage(message)
    chat_id = message.chat.id

    try:
        records = load_trade_records(chat_id)
        bundle = train_win_model(records)
        holdout = evaluate_on_holdout(bundle)
        save_model(bundle, chat_id, DATA_DIR)

        lines = [
            f"Model trained on {bundle['n_trades_trained_on']} trades "
            f"(most recent {holdout['n_holdout']} held out for testing).",
            "",
            f"Holdout accuracy (real signal): {holdout['accuracy']:.1%}",
        ]
        if holdout["auc"] is not None:
            lines.append(f"Holdout AUC: {holdout['auc']:.3f}  (0.5 = no better than random)")
        lines.append("")
        lines.append("Feature importance (from training data):")
        for name, imp in list(bundle["model"].feature_importance().items())[:8]:
            lines.append(f"  {name}: {imp:.2%}")
        lines.append("")
        lines.append("Run /predict for a hypothetical trade, or /predict_all / /backtest_model.")
        bot.send_message(chat_id, "\n".join(lines))
    except ValueError as e:
        bot.send_message(chat_id, str(e))
    except Exception as e:
        bot.send_message(chat_id, f"Error training model: {e}")


@bot.message_handler(commands=['backtest_model'])
def format_prediction_report(header_lines, results, footer_lines):
    """Shared formatter for /predict_all and /backtest_model.

    Sorts by model confidence (most confident calls first), marks
    each line ✅/❌, and puts a correct/incorrect count up top so the
    reader doesn't have to tally by hand.
    """
    correct = sum(1 for r in results if (r["predicted_prob"] >= 0.5) == r["actual_win"])
    total = len(results)

    ranked = sorted(results, key=lambda r: abs(r["predicted_prob"] - 0.5), reverse=True)

    lines = list(header_lines)
    lines.append(f"✅ {correct} correct   ❌ {total - correct} incorrect   ({total} total)")
    lines.append("Sorted by model confidence, most confident first.")
    lines.append("")

    for res in ranked:
        r = res["record"]
        asset = (r.get("asset") or "-").ljust(7)
        direction = (r.get("direction") or "-").ljust(5)
        actual = "WIN " if res["actual_win"] else "LOSS"
        predicted_label = "WIN " if res["predicted_prob"] >= 0.5 else "LOSS"
        mark = "✅" if predicted_label.strip() == actual.strip() else "❌"
        conf = res["predicted_prob"] if res["predicted_prob"] >= 0.5 else 1 - res["predicted_prob"]
        lines.append(
            f"{mark} {asset} {direction} pred {predicted_label} ({conf:.0%} conf)  actual {actual}"
        )

    lines.append("")
    lines.extend(footer_lines)
    return "\n".join(lines)


def send_long_message(chat_id, text):
    if len(text) > 4000:
        for i in range(0, len(text), 4000):
            bot.send_message(chat_id, text[i:i + 4000])
    else:
        bot.send_message(chat_id, text)


def handle_backtest_model(message):
    track_usage(message)
    chat_id = message.chat.id

    try:
        bundle = load_model(chat_id, DATA_DIR)
        holdout_records = bundle.get("holdout_records") or []
        if not holdout_records:
            bot.send_message(chat_id, "No holdout set stored on this model. Run /train_model again.")
            return

        results = predict_batch(bundle, holdout_records)
        holdout = evaluate_on_holdout(bundle)

        header = [f"Holdout backtest — {holdout['n_holdout']} trades the model never trained on:", ""]
        footer = [f"Holdout accuracy: {holdout['accuracy']:.1%}"]
        if holdout["auc"] is not None:
            footer.append(f"Holdout AUC: {holdout['auc']:.3f}")
        footer.append("Out-of-sample — the model never saw these trades during training.")

        text = format_prediction_report(header, results, footer)
        send_long_message(chat_id, text)

    except ValueError as e:
        bot.send_message(chat_id, str(e))
    except Exception as e:
        bot.send_message(chat_id, f"Error running backtest: {e}")


@bot.message_handler(commands=['predict'])
def handle_predict(message):
    track_usage(message)
    chat_id = message.chat.id
    msg = bot.send_message(
        chat_id,
        "Send one or more lines: `ASSET DIRECTION SESSION HH:MM`\n"
        "Example:\n"
        "EURUSD long London 09:15\n"
        "XAUUSD short Tokyo 22:00",
        parse_mode="Markdown"
    )
    bot.register_next_step_handler(msg, process_predict_step)


def process_predict_step(message):
    chat_id = message.chat.id
    lines = [l.strip() for l in message.text.strip().splitlines() if l.strip()]

    if not lines:
        bot.send_message(chat_id, "Empty input. Please run /predict again.")
        return

    try:
        bundle = load_model(chat_id, DATA_DIR)
    except ValueError as e:
        bot.send_message(chat_id, str(e))
        return

    results = []
    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 4:
            bot.send_message(
                chat_id,
                f"Invalid format on line {line_number}: `{line}`\nUse: `ASSET DIRECTION SESSION HH:MM`",
                parse_mode="Markdown"
            )
            return

        asset, direction, session, time_str = parts
        try:
            hh, mm = time_str.split(":")
            hour = int(hh)
            if not (0 <= hour <= 23 and 0 <= int(mm) <= 59):
                raise ValueError
        except ValueError:
            bot.send_message(chat_id, f"Invalid time on line {line_number}: '{time_str}', expected HH:MM.")
            return

        day_of_week = datetime_now_weekday()

        try:
            prob = predict_win_probability(bundle, asset.upper(), direction.lower(), session, hour, day_of_week)
        except Exception as e:
            bot.send_message(chat_id, f"Error on line {line_number}: {e}")
            return

        results.append((asset.upper(), direction.lower(), session, time_str, prob))

    lines_out = []
    for asset, direction, session, time_str, prob in results:
        lines_out.append(f"{asset} {direction} {session} {time_str} — {prob:.1%}")
    lines_out.append("")
    lines_out.append("Directional signal only, not a guarantee.")

    bot.send_message(chat_id, "\n".join(lines_out))


def datetime_now_weekday():
    from datetime import datetime
    return datetime.utcnow().weekday()


@bot.message_handler(commands=['predict_all'])
def handle_predict_all(message):
    track_usage(message)
    chat_id = message.chat.id

    try:
        records = load_trade_records(chat_id)
        bundle = load_model(chat_id, DATA_DIR)
        results = predict_batch(bundle, records)

        header = [f"Predictions for all {len(results)} stored trades (mix of train + holdout — not a clean metric):", ""]
        footer = ["(Real out-of-sample signal — run /backtest_model instead.)"]

        text = format_prediction_report(header, results, footer)
        send_long_message(chat_id, text)

    except ValueError as e:
        bot.send_message(chat_id, str(e))
    except Exception as e:
        bot.send_message(chat_id, f"Error running batch prediction: {e}")


if __name__ == "__main__":
    bot.set_my_commands([
        telebot.types.BotCommand("start", "Start the bot"),
        telebot.types.BotCommand("run", "Run Monte Carlo simulation"),
        telebot.types.BotCommand("add_trade", "Add one trade"),
        telebot.types.BotCommand("add_list", "Add a list of trades"),
        telebot.types.BotCommand("clear", "Clear all trades"),
        telebot.types.BotCommand("trades", "List trades"),
        telebot.types.BotCommand("help", "Help"),
        telebot.types.BotCommand("instructions", "What the ML predictions mean"),
        telebot.types.BotCommand("filter", "Filter by direction/asset"),
        telebot.types.BotCommand("import_mt5", "Import MT5 history"),
        telebot.types.BotCommand("train_model", "Train win-probability model"),
        telebot.types.BotCommand("backtest_model", "Score model on holdout (out-of-sample)"),
        telebot.types.BotCommand("predict", "Predict win probability for a trade"),
        telebot.types.BotCommand("predict_all", "Predict win probability for all stored trades"),
    ])
    print("Bot is listening for commands...")
    bot.infinity_polling()