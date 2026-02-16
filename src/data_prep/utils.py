"""
Shared utilities: data ingestion, rolling cutoff generation, and churn labels.
"""

from datetime import timedelta
from typing import List

import pandas as pd


def ingest_and_clean(path) -> pd.DataFrame:
    """Load raw Excel data and apply basic cleaning."""
    df = pd.read_excel(path)

    # Drop rows without a CustomerID — we can't attribute these to anyone
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)

    # Revenue = price × quantity (can be negative for returns)
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    # Cancellation invoices start with 'C' — we keep them for return_rate
    # but exclude them from value-based features
    df["is_cancellation"] = df["InvoiceNo"].astype(str).str.startswith("C")

    return df


def generate_cutoff_dates(
    df: pd.DataFrame,
    feature_window: int,
    churn_window: int,
    step: int,
) -> List[pd.Timestamp]:
    """
    Compute rolling cutoff dates from the dataset's date range.

    Constraints:
      - Earliest cutoff = min(InvoiceDate) + feature_window
        (need enough history to compute features)
      - Latest cutoff  = max(InvoiceDate) - churn_window
        (need enough future data to compute the churn label)
      - Cutoffs are spaced `step` days apart
    """
    date_min = df["InvoiceDate"].min()
    date_max = df["InvoiceDate"].max()

    earliest = date_min + timedelta(days=feature_window)
    latest = date_max - timedelta(days=churn_window)

    cutoffs = []
    current = earliest
    while current <= latest:
        cutoffs.append(pd.Timestamp(current))
        current += timedelta(days=step)

    return cutoffs


def build_churn_labels(
    df: pd.DataFrame, cutoff: pd.Timestamp, churn_window: int, feature_window: int
) -> pd.DataFrame:
    """
    30-day churn label for each customer at a given cutoff:
      churn = 1  →  ZERO purchases in [cutoff, cutoff + 30d)
      churn = 0  →  at least one purchase in that window

    Only customers who were active in the 90-day feature window before
    the cutoff are labelled (matching the feature population).
    """
    window_start = cutoff - timedelta(days=feature_window)
    window_end = cutoff + timedelta(days=churn_window)

    # Customers who DID purchase in the churn window
    post_purchases = df[
        (df["InvoiceDate"] >= cutoff)
        & (df["InvoiceDate"] < window_end)
        & (~df["is_cancellation"])
    ]
    active_in_window = set(post_purchases["CustomerID"].unique())

    # Population: customers active in the 90-day feature window
    feature_customers = df[
        (df["InvoiceDate"] >= window_start)
        & (df["InvoiceDate"] < cutoff)
        & (~df["is_cancellation"])
    ]["CustomerID"].unique()

    labels = pd.DataFrame({"CustomerID": feature_customers})
    labels["churn"] = (~labels["CustomerID"].isin(active_in_window)).astype(int)

    return labels
