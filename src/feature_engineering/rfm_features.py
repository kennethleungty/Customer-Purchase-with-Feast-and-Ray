"""
RFM (Recency-Frequency-Monetary) feature engineering.
"""

from datetime import timedelta

import pandas as pd


def build_rfm_features(
    df: pd.DataFrame, cutoff: pd.Timestamp, feature_window: int
) -> pd.DataFrame:
    """
    RFM + Tenure, one row per customer.

    Features are computed from the 90-day window [cutoff - 90d, cutoff):
      - recency_days:  days since most recent purchase in the window
      - frequency:     number of distinct orders in the window
      - monetary:      total spend in the window

    tenure_days is an exception — it uses ALL history up to the cutoff
    (days since the customer's very first purchase), since it's a customer
    attribute rather than a windowed metric.
    """
    window_start = cutoff - timedelta(days=feature_window)

    # Non-cancellation transactions within the feature window
    txn = df[
        (df["InvoiceDate"] >= window_start)
        & (df["InvoiceDate"] < cutoff)
        & (~df["is_cancellation"])
    ]

    if txn.empty:
        return pd.DataFrame(
            columns=["CustomerID", "recency_days", "frequency", "monetary", "tenure_days"]
        )

    agg = txn.groupby("CustomerID").agg(
        last_purchase=("InvoiceDate", "max"),
        frequency=("InvoiceNo", "nunique"),
        monetary=("Revenue", "sum"),
    )

    agg["recency_days"] = (cutoff - agg["last_purchase"]).dt.days

    # Tenure: all-time first purchase (not windowed)
    all_time = df[(df["InvoiceDate"] < cutoff) & (~df["is_cancellation"])]
    first_purchase = all_time.groupby("CustomerID")["InvoiceDate"].min()
    agg = agg.join(first_purchase.rename("first_purchase"))
    agg["tenure_days"] = (cutoff - agg["first_purchase"]).dt.days

    return agg[["recency_days", "frequency", "monetary", "tenure_days"]].reset_index()
