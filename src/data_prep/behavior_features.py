"""
Behavioral feature engineering.
"""

from datetime import timedelta

import pandas as pd


def build_behavior_features(
    df: pd.DataFrame, cutoff: pd.Timestamp, feature_window: int
) -> pd.DataFrame:
    """
    Behavioral features from the 90-day window [cutoff - 90d, cutoff):

      - avg_order_value:             mean revenue per order
      - avg_basket_size:             mean item count per order
      - n_unique_products:           distinct products bought
      - return_rate:                 share of orders that were cancellations
      - avg_days_between_purchases:  mean gap between consecutive orders
    """
    window_start = cutoff - timedelta(days=feature_window)

    # All transactions (including cancellations for return_rate) in the window
    pre = df[(df["InvoiceDate"] >= window_start) & (df["InvoiceDate"] < cutoff)]
    non_cancel = pre[~pre["is_cancellation"]]

    if non_cancel.empty:
        return pd.DataFrame(
            columns=[
                "CustomerID", "avg_order_value", "avg_basket_size",
                "n_unique_products", "return_rate", "avg_days_between_purchases",
            ]
        )

    # --- Per-order aggregation (one row per invoice) ---
    invoice_stats = (
        non_cancel.groupby(["CustomerID", "InvoiceNo"])
        .agg(
            order_revenue=("Revenue", "sum"),
            order_qty=("Quantity", "sum"),
            order_date=("InvoiceDate", "first"),
        )
        .reset_index()
    )

    # Average order value & basket size per customer
    order_avgs = invoice_stats.groupby("CustomerID").agg(
        avg_order_value=("order_revenue", "mean"),
        avg_basket_size=("order_qty", "mean"),
    )

    # Unique products purchased in the window
    unique_products = non_cancel.groupby("CustomerID")["StockCode"].nunique()
    unique_products.name = "n_unique_products"

    # Return rate: fraction of invoices in the window that are cancellations
    inv_flags = pre.groupby(["CustomerID", "InvoiceNo"])["is_cancellation"].first()
    return_rate = inv_flags.groupby("CustomerID").mean()
    return_rate.name = "return_rate"

    # Purchase cadence: average days between consecutive orders in the window
    sorted_dates = invoice_stats.sort_values("order_date")
    purchase_gaps = sorted_dates.groupby("CustomerID")["order_date"].apply(
        lambda dates: dates.diff().dt.days.mean()
    )
    purchase_gaps.name = "avg_days_between_purchases"

    # Combine all behavioral features
    behavior = (
        order_avgs.join(unique_products)
        .join(return_rate)
        .join(purchase_gaps)
        .reset_index()
    )

    # Customers with only 1 order in the window have no gap — fill with 0
    behavior["avg_days_between_purchases"] = behavior[
        "avg_days_between_purchases"
    ].fillna(0)

    return behavior
