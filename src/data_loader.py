import json
import numpy as np
from datetime import datetime, time as dtime, date as dtime_date
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def get_trades_path(chat_id):
    return DATA_DIR / f"trades_{chat_id}.jsonl"


def load_trade_records(chat_id):
    path = get_trades_path(chat_id)

    if not path.exists():
        raise ValueError("No trades yet. Use /add_trade or /add_list first.")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        raise ValueError("No trades yet. Use /add_trade or /add_list first.")

    return records


def load_trades(chat_id, asset=None, direction=None):
    records = load_trade_records(chat_id)

    if asset is not None:
        records = [r for r in records if r.get("asset", "").upper() == asset.upper()]
    if direction is not None:
        records = [r for r in records if r.get("direction", "").lower() == direction.lower()]

    if not records:
        raise ValueError("No trades match the given filters.")

    return np.array([r["r"] for r in records], dtype=float)


def trading_session(timestamp):
    t = datetime.fromisoformat(timestamp).time()

    if t >= dtime(18, 0) or t < dtime(2, 0):
        return "Tokyo"
    if dtime(2, 0) <= t < dtime(3, 0):
        return "Frankfurt"
    if dtime(3, 0) <= t < dtime(8, 0):
        return "London"
    if dtime(8, 0) <= t < dtime(12, 0):
        return "NewYork"
    return "Overlap"


def build_timestamp(time_str=None, date_str=None):
    if date_str:
        try:
            y, m, d = date_str.split(".")
            date_part = dtime_date(int(y), int(m), int(d))
        except ValueError:
            raise ValueError(f"invalid date '{date_str}', expected DD.MM.YYYY")
    else:
        date_part = datetime.utcnow().date()

    if time_str:
        try:
            hh, mm = time_str.split(":")
            time_part = dtime(int(hh), int(mm))
        except (ValueError, TypeError):
            raise ValueError(f"invalid time '{time_str}', expected HH:MM (24h)")
    else:
        time_part = datetime.utcnow().time().replace(microsecond=0) if not date_str else dtime(0, 0)

    return datetime.combine(date_part, time_part).isoformat(timespec="seconds")

def append_trade(chat_id, r, asset=None, direction=None, time_str=None):
    path = get_trades_path(chat_id)
    path.parent.mkdir(exist_ok=True)

    timestamp = build_timestamp(time_str)

    record = {
        "r": r,
        "asset": asset,
        "direction": direction,
        "timestamp": timestamp,
        "session": trading_session(timestamp),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def clear_trades(chat_id):
    path = get_trades_path(chat_id)
    path.parent.mkdir(exist_ok=True)
    path.write_text("")


def filter_trades(chat_id, direction=None, excluded_assets=None, session=None):
    records = load_trade_records(chat_id)

    if direction:
        records = [r for r in records if r.get("direction", "").lower() == direction.lower()]

    if excluded_assets:
        excluded = {a.upper() for a in excluded_assets}
        records = [r for r in records if r.get("asset", "").upper() not in excluded]

    if session:
        records = [r for r in records if r.get("session", "").lower() == session.lower()]

    if not records:
        raise ValueError("No trades match the given filters.")

    return np.array([r["r"] for r in records], dtype=float)