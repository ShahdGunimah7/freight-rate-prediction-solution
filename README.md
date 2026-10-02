# Freight Rate Prediction

Predicts the `posted_rate` of a truck load (Spotter ML Engineer assessment).
LightGBM on `log(rate)` with additive calendar effects. Validated with forward-in-time backtests.

| Backtest (clean labels) | MAPE | MAE |
|---|---|---|
| Fold A: train Jan–Apr → test May–Jun | 1.78% | $43 |
| Fold B: train Jan–Jul → test Aug–Sep | 1.67% | $38 |

Full write-up: [`reports/Freight_Rate_Report.pdf`](reports/Freight_Rate_Report.pdf)

## Quick start

Python 3.10+.

```bash
python -m pip install -r requirements.txt

python run_pipeline.py                  # clean, backtest, train, write predictions (~30 s)
python -m src.freight_rate.figures      # EDA + validation figures -> reports/figures/
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
python build_report.py                  # reports/Freight_Rate_Report.docx
python -m pytest -q                     # unit + output-format tests
```

The pipeline is deterministic: the same input gives byte-identical predictions.

## Outputs

| File | Description |
|---|---|
| `validation_predictions.csv` | 12,000 rows, `load_id,predicted_rate` |
| `data/december_chart_inputs.csv` | 31 December rows with `predicted_rate` filled |
| `scorer_results/candidate_december.png` | Chart produced by the provided `score.py` |
| `reports/metrics.json` | Data-quality counts, backtest metrics, feature importance |
| `reports/Freight_Rate_Report.pdf` / `.docx` | Report |

## Project structure

```
data/                      raw inputs (train_test, validation, template, december inputs)
src/freight_rate/
  config.py                paths, features, hyper-parameters, backtest folds
  data.py                  loading, cleaning, corrupted-label detection
  features.py              feature engineering (distance, geo, bearing, calendar)
  model.py                 LightGBM model (log target, interaction constraints, seed ensemble)
  pipeline.py              backtests -> final training -> predictions
  figures.py               report figures
  metrics.py               MAE / RMSE / MAPE / R2
tests/                     pytest suite
run_pipeline.py            entry point
build_report.py            builds the DOCX report
score.py                   provided scorer (unchanged)
```

## Approach in short

1. **Data quality.** 677 corrupted labels (rates ×0.17–0.45 or ×2.2–4.6) are detected with an iteratively refit robust log-linear model and excluded from training. Negative weights are sign errors (absolute value). Missing weight is filled with the equipment median. Missing `market_index` is filled with the same-day mean.
2. **Split.** Time-based, never random. Each fold trains up to the first month of a quarter and forecasts the next two months, mirroring Jan–Oct → Nov–Dec. Cleaning is re-fit inside each fold.
3. **Model.** LightGBM on `log(rate)`. Calendar features (quarter, days to quarter end, day of week, day of month) sit in separate interaction-constraint groups, so the quarter-end price ramp learned in Q1–Q3 carries over to Nov–Dec.
4. **Generalisation.** 8 validation cities never appear in training, so the model uses coordinates instead of city IDs. Holding out 8 cities in a backtest gave 1.71% vs 1.68% MAPE.
5. **December chart.** Daily market signals (`market_index`, `quote_signal`) are taken from `validation.csv` for each date.
