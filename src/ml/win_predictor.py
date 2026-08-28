import pickle
from pathlib import Path

from src.ml.boosting import Boosting
from src.ml.features import build_feature_matrix, encode_single

MIN_TRADES_TO_TRAIN = 30


def model_path(chat_id, data_dir):
    return Path(data_dir) / f"model_{chat_id}.pkl"


def train_win_model(records, n_estimators=50, learning_rate=0.15, max_depth=3):
    if len(records) < MIN_TRADES_TO_TRAIN:
        raise ValueError(
            f"Need at least {MIN_TRADES_TO_TRAIN} trades to train (have {len(records)}). "
            "Fewer than that and the model just memorizes noise."
        )

    X, y, feature_names, asset_vocab, session_vocab = build_feature_matrix(records)

    if len(set(y.tolist())) < 2:
        raise ValueError("All trades are the same outcome (all wins or all losses) — nothing to learn.")

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
        "n_trades_trained_on": len(records),
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