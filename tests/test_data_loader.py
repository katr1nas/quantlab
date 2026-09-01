import pytest

from src.data_loader import (
    append_trade,
    load_trades,
    load_trade_records,
    clear_trades,
    filter_trades,
    trading_session,
    get_trades_path,
)


def test_append_and_load_trades(tmp_data_dir):
    append_trade("u1", 1.5, "EURUSD", "long")
    append_trade("u1", -1.0, "XAUUSD", "short")

    trades = load_trades("u1")
    assert list(trades) == [1.5, -1.0]


def test_load_trades_missing_file_raises(tmp_data_dir):
    with pytest.raises(ValueError):
        load_trades("nobody")


def test_load_trade_records_has_expected_fields(tmp_data_dir):
    append_trade("u1", 1.0, "EURUSD", "long", "09:15")
    records = load_trade_records("u1")

    assert len(records) == 1
    r = records[0]
    assert r["r"] == 1.0
    assert r["asset"] == "EURUSD"
    assert r["direction"] == "long"
    assert "timestamp" in r
    assert "session" in r


def test_clear_trades_empties_file(tmp_data_dir):
    append_trade("u1", 1.0, "EURUSD", "long")
    clear_trades("u1")
    with pytest.raises(ValueError):
        load_trade_records("u1")


def test_filter_trades_by_direction(tmp_data_dir):
    append_trade("u1", 1.0, "EURUSD", "long")
    append_trade("u1", -1.0, "EURUSD", "short")
    append_trade("u1", 2.0, "GBPUSD", "long")

    longs = filter_trades("u1", direction="long")
    assert list(longs) == [1.0, 2.0]


def test_filter_trades_excludes_assets(tmp_data_dir):
    append_trade("u1", 1.0, "EURUSD", "long")
    append_trade("u1", 2.0, "XAUUSD", "long")

    filtered = filter_trades("u1", excluded_assets=["XAUUSD"])
    assert list(filtered) == [1.0]


def test_filter_trades_no_match_raises(tmp_data_dir):
    append_trade("u1", 1.0, "EURUSD", "long")
    with pytest.raises(ValueError):
        filter_trades("u1", direction="short")


def test_trading_session_buckets():
    assert trading_session("2026-08-31T05:00:00") == "London"
    assert trading_session("2026-08-31T09:00:00") in ("NewYork", "New York")  # tolerate either naming
    assert trading_session("2026-08-31T23:30:00") == "Tokyo"
    assert trading_session("2026-08-31T02:30:00") == "Frankfurt"


def test_filter_trades_by_session(tmp_data_dir):
    append_trade("u1", 1.0, "EURUSD", "long", "05:00")   # London
    append_trade("u1", -1.0, "EURUSD", "short", "23:30")  # Tokyo

    london_only = filter_trades("u1", session="London")
    assert list(london_only) == [1.0]


def test_get_trades_path_is_per_chat_id(tmp_data_dir):
    p1 = get_trades_path("u1")
    p2 = get_trades_path("u2")
    assert p1 != p2
    assert "u1" in str(p1)
    assert "u2" in str(p2)


# --- Date support tests ---
# Written against the DD.MM.YYYY dot-separated format you implemented.
# If your final build_timestamp uses a different format (e.g. YYYY-MM-DD),
# update the date strings below to match before running.

def test_append_trade_with_explicit_date(tmp_data_dir):
    append_trade("u1", 1.0, "EURUSD", "long", "09:15", "15.06.2026")
    records = load_trade_records("u1")
    assert records[0]["timestamp"].startswith("2026-06-15")


def test_holdout_split_respects_real_dates(tmp_data_dir):
    from src.ml.win_predictor import split_train_holdout

    append_trade("u1", 1.0, "EURUSD", "long", "10:00", "01.06.2026")
    append_trade("u1", 1.0, "EURUSD", "long", "10:00", "15.06.2026")
    append_trade("u1", 1.0, "EURUSD", "long", "10:00", "30.06.2026")

    records = load_trade_records("u1")
    train, holdout = split_train_holdout(records, holdout_fraction=0.34)

    assert holdout[0]["timestamp"].startswith("2026-06-30")