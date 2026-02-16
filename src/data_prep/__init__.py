"""
Data preparation package — re-exports public API so that existing imports
like `from src.data_prep import ingest_and_clean` continue to work.
"""

from src.data_prep.behavior_features import build_behavior_features
from src.data_prep.rfm_features import build_rfm_features
from src.data_prep.utils import build_churn_labels, generate_cutoff_dates, ingest_and_clean

__all__ = [
    "ingest_and_clean",
    "generate_cutoff_dates",
    "build_rfm_features",
    "build_behavior_features",
    "build_churn_labels",
]
