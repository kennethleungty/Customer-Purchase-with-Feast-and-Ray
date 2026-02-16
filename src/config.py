"""
Centralized configuration for the purchase propensity pipeline.
All shared constants (file paths, window sizes, model params) live here
so they can be changed in one place.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FEATURE_STORE_DIR = ROOT_DIR / "feature_store"
FEATURE_DATA_DIR = FEATURE_STORE_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"

RAW_DATA_PATH = DATA_DIR / "input" / "Online Retail.xlsx"

# Parquet outputs consumed by Feast as offline data sources
RFM_FEATURES_PATH = FEATURE_DATA_DIR / "customer_rfm_features.parquet"
BEHAVIOR_FEATURES_PATH = FEATURE_DATA_DIR / "customer_behavior_features.parquet"

# ---------------------------------------------------------------------------
# Rolling window & purchase label definition
# ---------------------------------------------------------------------------
# For each cutoff date C:
#   - Features are computed from the FEATURE_WINDOW_DAYS before C
#     i.e. transactions in [C - 90d, C)
#   - Purchase label is computed from the PURCHASE_WINDOW_DAYS after C
#     i.e. purchased = 1 if at least one purchase in [C, C + 30d)
#   - Cutoff dates are spaced ROLLING_STEP_DAYS apart
FEATURE_WINDOW_DAYS = 90
PURCHASE_WINDOW_DAYS = 30
ROLLING_STEP_DAYS = 30

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_PATH = MODEL_DIR / "xgb_purchase_model.json"
PREDICTIONS_PATH = MODEL_DIR / "predictions.parquet"

RANDOM_STATE = 42

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "random_state": RANDOM_STATE,
}

# ---------------------------------------------------------------------------
# Feature lists (must match the Feast FeatureView schemas in definitions.py)
# ---------------------------------------------------------------------------
RFM_FEATURES = [
    "recency_days",
    "frequency",
    "monetary",
    "tenure_days",
]

BEHAVIOR_FEATURES = [
    "avg_order_value",
    "avg_basket_size",
    "n_unique_products",
    "return_rate",
    "avg_days_between_purchases",
]

ALL_FEATURES = RFM_FEATURES + BEHAVIOR_FEATURES
