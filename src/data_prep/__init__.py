"""
Data preparation package — ingestion, cutoff generation, and churn labels.
"""

from src.data_prep.cutoffs import generate_cutoff_dates
from src.data_prep.ingestion import ingest_and_clean
from src.data_prep.labels import build_churn_labels

__all__ = [
    "ingest_and_clean",
    "generate_cutoff_dates",
    "build_churn_labels",
]
