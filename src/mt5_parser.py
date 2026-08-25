
import pandas as pd


def inspect_tables(html_path):
    tables = pd.read_html(html_path)
    for i, t in enumerate(tables):
        print(f"[{i}] shape={t.shape} columns={list(t.columns)}")
    return tables


def _find_deals_table(tables):
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if "profit" in cols and "symbol" in cols and "direction" in cols:
            return t
    raise ValueError(
        "Could not find a 'Deals' table with Profit/Symbol/Direction columns. "
        "Run inspect_tables() and check the actual headers in your export."
    )


def parse_mt5_html(html_path, account_balance, risk_pct):
    if account_balance <= 0:
        raise ValueError("account_balance must be positive")
    if not (0 < risk_pct < 1):
        raise ValueError("risk_pct must be a fraction between 0 and 1, e.g. 0.01 for 1%")

    risk_amount = account_balance * risk_pct

    tables = pd.read_html(html_path)
    deals = _find_deals_table(tables)
    deals.columns = [str(c).strip().lower() for c in deals.columns]

    closed = deals[deals["direction"].astype(str).str.strip().str.lower() == "out"].copy()

    trades = []
    for _, row in closed.iterrows():
        symbol = str(row.get("symbol", "")).strip().upper()
        deal_type = str(row.get("type", "")).strip().lower()
        try:
            profit = float(row.get("profit", 0))
        except (TypeError, ValueError):
            continue

        if not symbol or deal_type not in ("buy", "sell"):
            continue

        # closing deal type is the OPPOSITE of the position's original direction
        direction = "short" if deal_type == "buy" else "long"

        r = profit / risk_amount
        trades.append({"r": r, "asset": symbol, "direction": direction})

    if not trades:
        raise ValueError("No closed trades found in the report (no 'out' rows matched).")

    return trades


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python -m src.mt5_parser <report.html> <account_balance> <risk_pct>")
        print("Example: python -m src.mt5_parser report.html 50000 0.01")
        sys.exit(1)

    path, balance, risk = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    result = parse_mt5_html(path, balance, risk)
    print(f"Parsed {len(result)} closed trades.")
    for t in result[:10]:
        print(t)