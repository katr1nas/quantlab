from datetime import datetime

import numpy as np


def build_feature_matrix(records, asset_vocab=None, session_vocab=None):
    if asset_vocab is None:
        asset_vocab = sorted({r.get("asset") or "UNKNOWN" for r in records})
    if session_vocab is None:
        session_vocab = sorted({r.get("session") or "UNKNOWN" for r in records})

    feature_names = (
        [f"asset={a}" for a in asset_vocab]
        + ["direction"]
        + [f"session={s}" for s in session_vocab]
        + ["hour", "day_of_week"]
    )

    rows = []
    labels = []
    for r in records:
        row = _encode_record(r, asset_vocab, session_vocab)
        rows.append(row)
        labels.append(1 if r["r"] > 0 else 0)

    X = np.array(rows, dtype=float)
    y = np.array(labels, dtype=float)
    return X, y, feature_names, asset_vocab, session_vocab


def _encode_record(record, asset_vocab, session_vocab):
    asset = record.get("asset") or "UNKNOWN"
    session = record.get("session") or "UNKNOWN"
    direction = 1.0 if (record.get("direction") or "").lower() == "long" else 0.0

    asset_onehot = [1.0 if asset == a else 0.0 for a in asset_vocab]
    session_onehot = [1.0 if session == s else 0.0 for s in session_vocab]

    ts = record.get("timestamp")
    if ts:
        dt = datetime.fromisoformat(ts)
        hour, dow = float(dt.hour), float(dt.weekday())
    else:
        hour, dow = 0.0, 0.0

    return asset_onehot + [direction] + session_onehot + [hour, dow]


def encode_single(asset, direction, session, hour, day_of_week, asset_vocab, session_vocab):
    record = {
        "asset": asset,
        "direction": direction,
        "session": session,
        "timestamp": None,
    }
    row = _encode_record(record, asset_vocab, session_vocab)
    row[-2] = float(hour)
    row[-1] = float(day_of_week)
    return np.array([row], dtype=float)