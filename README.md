# QuantLab

QuantLab is a personal research environment for testing trading hypotheses with historical data, statistics, validation tools, and visual reports.

## Strategy return research pipeline

The root-level `research.py` script provides one entry point for a repeatable train/OOS report from a CSV file of periodic **simple returns**.

### Input format

Your CSV should contain a date column and a return column. By default, the column names are `date` and `return`.

```csv
date,return
2024-01-02,0.004
2024-01-03,-0.012
2024-01-04,0.007
```

Returns must be decimals (for example, `0.01` means +1%), not percentage values such as `1`.

### Run the pipeline

From the repository root, with dependencies installed:

```bash
python research.py data/strategies/mean_reversion.csv --name "Mean Reversion"
```

Useful options:

```bash
python research.py data/strategies/mean_reversion.csv \
  --name "Mean Reversion" \
  --return-column return \
  --date-column date \
  --train-size 0.7 \
  --periods-per-year 252
```

Use `--periods-per-year 252` for daily observations or `12` for monthly observations. Choose a value consistent with the frequency of the return series.

### Outputs

The report is written under `reports/research/<strategy-name>/`:

- `metrics.csv` — train, OOS, and full-sample metrics.
- `performance.png` — equity curve and drawdown, with the OOS boundary marked.
- `summary.txt` — date ranges, sample sizes, settings, and metrics.

The pipeline checks required columns, invalid dates, duplicate timestamps, missing/non-numeric values, non-finite values, and impossible simple returns at or below -100%.

## Research discipline

- The split is chronological and does not shuffle observations.
- The train/OOS split of an already-produced return series is only a valid holdout if the strategy was not fitted or tuned using the OOS period. For strategies fitted on prices/features, rerun the strategy independently using train-only fitting and a genuinely untouched OOS period.
- Do not repeatedly tune a strategy based on the same OOS results and continue to call that period untouched OOS.
- Metrics assume the input is a correctly aligned series of periodic strategy returns. Transaction costs, slippage, financing, and position-sizing assumptions must already be reflected if they are relevant to the hypothesis.
- Lookahead validation requires the relevant price/features and feature-generation function; it cannot be proven from a returns-only CSV.

## Existing modules

- `src/metrics.py` — return and trade statistics.
- `src/trade_loader.py` — load strategy return CSVs from `data/strategies/`.
- `visualization/plots.py` — performance and distribution charts.
- `validation/` — validation utilities.

The pipeline orchestrates the existing metric and plotting modules rather than duplicating their core logic.
