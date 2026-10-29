import numpy as np
import pandas as pd
import pytest

from research import load_returns, split_returns


def test_load_returns_sorts_by_date(tmp_path):
    path = tmp_path / "returns.csv"
    pd.DataFrame({
        "date": ["2024-01-03", "2024-01-01", "2024-01-02"],
        "return": [0.03, 0.01, -0.02],
    }).to_csv(path, index=False)

    returns = load_returns(path)

    assert returns.index.is_monotonic_increasing
    assert returns.tolist() == [0.01, -0.02, 0.03]


def test_load_returns_rejects_duplicate_dates(tmp_path):
    path = tmp_path / "returns.csv"
    pd.DataFrame({
        "date": ["2024-01-01", "2024-01-01"],
        "return": [0.01, 0.02],
    }).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Duplicate dates"):
        load_returns(path)


def test_load_returns_rejects_returns_at_or_below_minus_one(tmp_path):
    path = tmp_path / "returns.csv"
    pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        "return": [0.01, -1.0, 0.02, 0.01],
    }).to_csv(path, index=False)

    with pytest.raises(ValueError, match="greater than -1"):
        load_returns(path)


def test_split_returns_is_chronological_and_disjoint():
    returns = pd.Series(
        np.arange(10, dtype=float) / 100,
        index=pd.date_range("2024-01-01", periods=10),
    )

    train, oos = split_returns(returns, train_size=0.7)

    assert len(train) == 7
    assert len(oos) == 3
    assert train.index.max() < oos.index.min()
    assert train.iloc[-1] == returns.iloc[6]
    assert oos.iloc[0] == returns.iloc[7]


def test_split_returns_rejects_invalid_fraction():
    returns = pd.Series(
        np.zeros(10),
        index=pd.date_range("2024-01-01", periods=10),
    )

    with pytest.raises(ValueError, match="train_size"):
        split_returns(returns, train_size=0.4)
