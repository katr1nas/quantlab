"""Run a repeatable train/OOS research report for a strategy-return CSV.

Example:
    python research.py data/strategies/mean_reversion.csv --name "Mean Reversion"

The input CSV must contain a `date` column and a column of simple periodic returns
(default: `return`). Returns must be decimal returns (e.g. 0.02 for +2%).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import calculate_metrics
from visualization.plots import plot_performance


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research"


def load_returns(
    csv_path: str | Path,
    return_column: str = "return",
    date_column: str = "date",
) -> pd.Series:
    """Load and validate a dated series of simple returns."""
    path = Path(csv_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Returns CSV not found: {path}")

    frame = pd.read_csv(path)
    missing = {date_column, return_column} - set(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(sorted(missing))}")

    dates = pd.to_datetime(frame[date_column], errors="coerce")
    if dates.isna().any():
        bad_rows = (np.flatnonzero(dates.isna().to_numpy()) + 2).tolist()[:5]
        raise ValueError(f"Invalid or missing dates in CSV row(s): {bad_rows}")
    if dates.duplicated().any():
        examples = dates[dates.duplicated(keep=False)].astype(str).unique()[:5]
        raise ValueError(f"Duplicate dates found: {', '.join(examples)}")

    values = pd.to_numeric(frame[return_column], errors="coerce")
    if values.isna().any():
        bad_rows = (np.flatnonzero(values.isna().to_numpy()) + 2).tolist()[:5]
        raise ValueError(f"Missing or non-numeric returns in CSV row(s): {bad_rows}")

    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Returns contain infinite or non-finite values.")
    if (array <= -1).any():
        raise ValueError("Simple returns must be greater than -1 (−100%).")

    returns = pd.Series(array, index=pd.DatetimeIndex(dates, name=date_column), name=return_column)
    returns = returns.sort_index()
    if len(returns) < 4:
        raise ValueError("At least 4 observations are required for a train/OOS split.")
    return returns


def split_returns(returns: pd.Series, train_size: float = 0.7) -> tuple[pd.Series, pd.Series]:
    """Chronologically split returns without shuffling."""
    if not 0.5 <= train_size <= 0.9:
        raise ValueError("train_size must be between 0.5 and 0.9.")
    split_at = int(len(returns) * train_size)
    if split_at < 2 or len(returns) - split_at < 2:
        raise ValueError("The train and OOS sections must each contain at least 2 observations.")
    return returns.iloc[:split_at].copy(), returns.iloc[split_at:].copy()


def calculate_report_metrics(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Calculate the project's existing metrics for train, OOS, and full history."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer.")
    rows = {
        "train": calculate_metrics(returns[0], periods_per_year),
        "oos": calculate_metrics(returns[1], periods_per_year),
        "full": calculate_metrics(returns[2], periods_per_year),
    }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("period")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "strategy"


def run_research(
    csv_path: str | Path,
    name: str | None = None,
    return_column: str = "return",
    date_column: str = "date",
    train_size: float = 0.7,
    periods_per_year: int = 252,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path | pd.DataFrame]:
    """Run validation, train/OOS analysis, plots, and export a compact report."""
    returns = load_returns(csv_path, return_column, date_column)
    train, oos = split_returns(returns, train_size)
    strategy_name = name or Path(csv_path).stem

    metrics = calculate_report_metrics(
        (train, oos, returns),
        periods_per_year=periods_per_year,
    )

    destination = Path(output_dir)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination = destination / _slug(strategy_name)
    destination.mkdir(parents=True, exist_ok=True)

    metrics_path = destination / "metrics.csv"
    plot_path = destination / "performance.png"
    summary_path = destination / "summary.txt"
    metrics.to_csv(metrics_path, float_format="%.8g")

    fig, (ax_equity, _) = plot_performance(
        returns,
        title=f"{strategy_name}: Train / OOS",
        save_path=plot_path,
    )
    boundary = oos.index[0]
    ax_equity.axvline(boundary, color="#7C3AED", linestyle="--", linewidth=1.5)
    ax_equity.text(
        boundary, 0.04, "OOS starts",
        transform=ax_equity.get_xaxis_transform(),
        rotation=90, va="bottom", ha="right",
        color="#7C3AED", fontsize=9,
    )
    fig.savefig(plot_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    split_line = (
        f"Train: {train.index.min().date()} to {train.index.max().date()} "
        f"({len(train)} observations)\n"
        f"OOS:   {oos.index.min().date()} to {oos.index.max().date()} "
        f"({len(oos)} observations)\n"
    )
    summary = (
        f"Research report: {strategy_name}\n"
        f"{'=' * (16 + len(strategy_name))}\n"
        f"Input: {Path(csv_path)}\n"
        f"Return column: {return_column}\n"
        f"Train fraction: {train_size:.0%}\n"
        f"Annualization periods: {periods_per_year}\n"
        f"Total observations: {len(returns)}\n"
        f"{split_line}\n"
        "Metrics (train and OOS are calculated independently):\n"
        f"{metrics.to_string(float_format=lambda x: f'{x:.4f}')}\n\n"
        "Interpretation note: OOS results are a historical holdout, not a guarantee "
        "of future performance. Do not use OOS results to tune the strategy and then "
        "continue treating the same period as untouched OOS.\n"
    )
    summary_path.write_text(summary, encoding="utf-8")

    print(summary)
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved chart:   {plot_path}")
    print(f"Saved summary: {summary_path}")

    return {
        "metrics": metrics,
        "metrics_path": metrics_path,
        "plot_path": plot_path,
        "summary_path": summary_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a QuantLab train/OOS research report.")
    parser.add_argument("csv", help="Path to a strategy-return CSV")
    parser.add_argument("--name", help="Strategy name shown in the report")
    parser.add_argument("--return-column", default="return", help="CSV column containing simple returns")
    parser.add_argument("--date-column", default="date", help="CSV column containing observation dates")
    parser.add_argument("--train-size", type=float, default=0.7, help="Fraction assigned to train (0.5–0.9)")
    parser.add_argument("--periods-per-year", type=int, default=252, help="Annualization periods (252 for daily data)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for generated reports")
    args = parser.parse_args()

    run_research(
        csv_path=args.csv,
        name=args.name,
        return_column=args.return_column,
        date_column=args.date_column,
        train_size=args.train_size,
        periods_per_year=args.periods_per_year,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
