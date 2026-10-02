"""Central configuration: paths, feature lists and model hyper-parameters."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"
PREDICTIONS_PATH = ROOT / "validation_predictions.csv"

TARGET = "posted_rate"
RANDOM_SEED = 7

# --- Data cleaning -----------------------------------------------------------
# A label is treated as corrupted when it is more than exp(0.4) ~ 1.5x away from
# a robust log-linear fit. Clean residuals have a std of ~0.04, while corrupted
# labels sit at roughly x0.2-0.35 or x2-4, so there is a wide empty gap.
OUTLIER_LOG_THRESHOLD = 0.4

EQUIPMENT_CODES = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}

# --- Features ----------------------------------------------------------------
LOAD_FEATURES = [
    "log_distance",
    "equipment_code",
    "weight",
    "market_index",
    "quote_signal",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "bearing",
]
# Calendar features are kept in their own interaction groups so their effect is
# additive in log space (i.e. multiplicative on the rate). This is what lets the
# quarter-end ramp learned in Q1-Q3 transfer to Nov/Dec (end of Q4).
TIME_FEATURES = ["quarter", "days_to_quarter_end", "day_of_week", "day_of_month"]
FEATURES = LOAD_FEATURES + TIME_FEATURES
INTERACTION_GROUPS = [LOAD_FEATURES] + [[f] for f in TIME_FEATURES]

# --- Model -------------------------------------------------------------------
LGB_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.03,
    "num_leaves": 4,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
}
NUM_BOOST_ROUND = 3000
ENSEMBLE_SEEDS = [7, 11, 23]

# --- Validation: time-based backtests ----------------------------------------
# Each fold mirrors the real task: the model sees the first month of a quarter
# and must forecast the remaining two months (Jan-Oct -> Nov-Dec).
BACKTEST_FOLDS = [
    {"name": "Fold A", "train_end": "2025-04-30", "test_start": "2025-05-01", "test_end": "2025-06-30"},
    {"name": "Fold B", "train_end": "2025-07-31", "test_start": "2025-08-01", "test_end": "2025-09-30"},
]
