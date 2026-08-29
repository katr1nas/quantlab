import pickle
from pathlib import Path

from sklearn.metrics import roc_auc_score

from src.ml.boosting import Boosting
from src.ml.features import build_feature_matrix, encode_single

MIN_TRADES_TO_TRAIN = 150


def model_path(chat_id, data_dir):
    return Path(data_dir) / f"model_{chat_id}.pkl"


def split_train_holdout(records, holdout_fraction=0.3):
    sorted_records = sorted(records, key=lambda r: r.get("timestamp") or "")
    n_holdout = max(1, int(len(sorted_records) * holdout_fraction))
    train_records = sorted_records[:-n_holdout]
    holdout_records = sorted_records[-n_holdout:]
    return train_records, holdout_records


def train_win_model(records, holdout_fraction=0.3, n_estimators=50, learning_rate=0.15, max_depth=3):
    if len(records) < MIN_TRADES_TO_TRAIN:
        raise ValueError(
            f"Need at least {MIN_TRADES_TO_TRAIN} trades to train (have {len(records)}). "
            "Fewer than that and the model just memorizes noise."
        )

    train_records, holdout_records = split_train_holdout(records, holdout_fraction)

    if len(train_records) < MIN_TRADES_TO_TRAIN * (1 - holdout_fraction):
        raise ValueError(
            f"After holding out {holdout_fraction:.0%} for testing, only {len(train_records)} "
            "trades remain for training — not enough. Add more trades."
        )

    X, y, feature_names, asset_vocab, session_vocab = build_feature_matrix(train_records)

    if len(set(y.tolist())) < 2:
        raise ValueError("All training trades are the same outcome (all wins or all losses) — nothing to learn.")

    model = Boosting(
        base_model_params={"max_depth": max_depth},
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        use_bootstrap=True,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        random_state=42,
    )
    model.fit(X, y, feature_names=feature_names)

    return {
        "model": model,
        "asset_vocab": asset_vocab,
        "session_vocab": session_vocab,
        "n_trades_trained_on": len(train_records),
        "holdout_records": holdout_records,
    }


def evaluate_on_holdout(bundle):
    holdout_records = bundle.get("holdout_records") or []
    if not holdout_records:
        raise ValueError("No holdout set stored on this model. Retrain to get one.")

    results = predict_batch(bundle, holdout_records)
    y_true = [1 if r["actual_win"] else 0 for r in results]
    y_prob = [r["predicted_prob"] for r in results]

    correct = sum(1 for yt, yp in zip(y_true, y_prob) if (yp >= 0.5) == bool(yt))
    accuracy = correct / len(results)

    auc = None
    if len(set(y_true)) == 2:
        auc = roc_auc_score(y_true, y_prob)

    return {
        "n_holdout": len(results),
        "accuracy": accuracy,
        "auc": auc,
    }


def save_model(bundle, chat_id, data_dir):
    path = model_path(chat_id, data_dir)
    path.parent.mkdir(exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_model(chat_id, data_dir):
    path = model_path(chat_id, data_dir)
    if not path.exists():
        raise ValueError("No trained model yet. Run /train_model first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_win_probability(bundle, asset, direction, session, hour, day_of_week):
    X = encode_single(
        asset, direction, session, hour, day_of_week,
        bundle["asset_vocab"], bundle["session_vocab"],
    )
    proba = bundle["model"].predict_proba(X)
    return float(proba[0, 1])


def predict_batch(bundle, records):
    X, y, _, _, _ = build_feature_matrix(
        records,
        asset_vocab=bundle["asset_vocab"],
        session_vocab=bundle["session_vocab"],
    )
    probs = bundle["model"].predict_proba(X)[:, 1]

    results = []
    for record, prob, actual in zip(records, probs, y):
        results.append({
            "record": record,
            "predicted_prob": float(prob),
            "actual_win": bool(actual),
        })
    return results