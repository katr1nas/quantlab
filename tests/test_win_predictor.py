import random

import pytest

from src.data_loader import append_trade, load_trade_records
from src.ml.win_predictor import (
    train_win_model,
    split_train_holdout,
    evaluate_on_holdout,
    predict_win_probability,
    predict_batch,
    save_model,
    load_model,
    MIN_TRADES_TO_TRAIN,
)


def _load_random_trades(chat_id, n=200, seed=1):
    random.seed(seed)
    assets = ["EURUSD", "XAUUSD", "GBPUSD", "USDJPY", "NAS100"]
    for _ in range(n):
        r = round(random.choice([random.uniform(0.3, 3.0), -random.uniform(0.3, 1.2)]), 2)
        hh, mm = random.randint(0, 23), random.randint(0, 59)
        append_trade(chat_id, r, random.choice(assets), random.choice(["long", "short"]), f"{hh:02d}:{mm:02d}")
    return load_trade_records(chat_id)


def test_train_below_floor_raises(tmp_data_dir):
    records = _load_random_trades("u1", n=MIN_TRADES_TO_TRAIN - 1)
    with pytest.raises(ValueError):
        train_win_model(records)


def test_train_at_floor_succeeds(tmp_data_dir):
    records = _load_random_trades("u1", n=MIN_TRADES_TO_TRAIN + 50)
    bundle = train_win_model(records)
    assert bundle["model"] is not None
    assert bundle["n_trades_trained_on"] > 0
    assert len(bundle["holdout_records"]) > 0


def test_split_train_holdout_sizes():
    records = [{"timestamp": f"2026-01-{d:02d}T00:00:00", "r": 1, "asset": "EURUSD",
                "direction": "long", "session": "London"} for d in range(1, 101)]
    train, holdout = split_train_holdout(records, holdout_fraction=0.3)
    assert len(holdout) == 30
    assert len(train) == 70


def test_split_train_holdout_is_chronological():
    records = [{"timestamp": f"2026-01-{d:02d}T00:00:00", "r": 1, "asset": "EURUSD",
                "direction": "long", "session": "London"} for d in range(1, 11)]
    train, holdout = split_train_holdout(records, holdout_fraction=0.3)
    # holdout must be the latest dates, train the earliest
    assert max(r["timestamp"] for r in train) < min(r["timestamp"] for r in holdout)


def test_evaluate_on_holdout_returns_expected_keys(tmp_data_dir):
    records = _load_random_trades("u1", n=MIN_TRADES_TO_TRAIN + 50)
    bundle = train_win_model(records)
    result = evaluate_on_holdout(bundle)
    assert "n_holdout" in result
    assert "accuracy" in result
    assert 0.0 <= result["accuracy"] <= 1.0


def test_predict_win_probability_in_range(tmp_data_dir):
    records = _load_random_trades("u1", n=MIN_TRADES_TO_TRAIN + 50)
    bundle = train_win_model(records)
    prob = predict_win_probability(bundle, "EURUSD", "long", "London", 9, 2)
    assert 0.0 <= prob <= 1.0


def test_predict_batch_matches_record_count(tmp_data_dir):
    records = _load_random_trades("u1", n=MIN_TRADES_TO_TRAIN + 50)
    bundle = train_win_model(records)
    results = predict_batch(bundle, bundle["holdout_records"])
    assert len(results) == len(bundle["holdout_records"])
    for res in results:
        assert 0.0 <= res["predicted_prob"] <= 1.0
        assert isinstance(res["actual_win"], bool)


def test_save_and_load_model_roundtrip(tmp_data_dir):
    records = _load_random_trades("u1", n=MIN_TRADES_TO_TRAIN + 50)
    bundle = train_win_model(records)
    save_model(bundle, "u1", tmp_data_dir)

    loaded = load_model("u1", tmp_data_dir)
    prob_before = predict_win_probability(bundle, "EURUSD", "long", "London", 9, 2)
    prob_after = predict_win_probability(loaded, "EURUSD", "long", "London", 9, 2)
    assert prob_before == pytest.approx(prob_after)


def test_load_model_missing_raises(tmp_data_dir):
    with pytest.raises(ValueError):
        load_model("nobody", tmp_data_dir)


def test_model_detects_strong_deterministic_signal(tmp_data_dir):
    """Positive control: if win outcome is a deterministic function of
    direction, the model must separate it near-perfectly on holdout.
    Confirms the pipeline can learn real signal, not just report noise as noise.
    """
    random.seed(3)
    for i in range(300):
        direction = random.choice(["long", "short"])
        is_win = direction == "long"  # deterministic, no noise
        r = 1.0 if is_win else -1.0
        hh, mm = random.randint(0, 23), random.randint(0, 59)
        dd = 1 + (i % 28)
        append_trade("u1", r, "EURUSD", direction, f"{hh:02d}:{mm:02d}", f"{dd:02d}.06.2026")

    records = load_trade_records("u1")
    bundle = train_win_model(records)
    result = evaluate_on_holdout(bundle)

    assert result["accuracy"] > 0.9
    assert result["auc"] is None or result["auc"] > 0.9