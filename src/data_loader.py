import json
import numpy as np
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


def append_trade(chat_id, r, asset=None, direction=None):
    path = get_trades_path(chat_id)
    path.parent.mkdir(exist_ok=True)

    record = {"r": r, "asset": asset, "direction": direction}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def clear_trades(chat_id):
    path = get_trades_path(chat_id)
    path.parent.mkdir(exist_ok=True)
    path.write_text("")

def filter_trades(chat_id, direction=None, excluded_assets=None):
    records = load_trade_records(chat_id)

    if direction:
        records = [r for r in records if r.get("direction", "").lower() == direction.lower()]

    if excluded_assets:
        excluded = {a.upper() for a in excluded_assets}
        records = [r for r in records if r.get("asset", "").upper() not in excluded]

    if not records:
        raise ValueError("No trades match the given filters.")

    return np.array([r["r"] for r in records], dtype=float)

trades = [
    {"r": 1.5, "asset": "EURUSD", "direction": "long"},
    {"r": -1.0, "asset": "XAUUSD", "direction": "short"},
    {"r": 0.5, "asset": "EURUSD", "direction": "long"},
]

print(filter_trades(trades, direction="long"))
    