"""
Churn label generation.
"""

from datetime import timedelta

import pandas as pd


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
