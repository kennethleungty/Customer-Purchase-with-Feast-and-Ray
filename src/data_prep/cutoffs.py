"""
Rolling cutoff date generation.
"""

from datetime import timedelta
from typing import List

import pandas as pd


def generate_cutoff_dates(
    df: pd.DataFrame,
    feature_window: int,
    purchase_window: int,
    step: int,
) -> List[pd.Timestamp]:
    """
    Compute rolling cutoff dates from the dataset's date range.

    Constraints:
      - Earliest cutoff = min(InvoiceDate) + feature_window
        (need enough history to compute features)
      - Latest cutoff  = max(InvoiceDate) - purchase_window
        (need enough future data to compute the purchase label)
      - Cutoffs are spaced `step` days apart
    """
    date_min = df["InvoiceDate"].min()
    date_max = df["InvoiceDate"].max()

    earliest = date_min + timedelta(days=feature_window)
    latest = date_max - timedelta(days=purchase_window)

    cutoffs = []
    current = earliest
    while current <= latest:
        cutoffs.append(pd.Timestamp(current))
        current += timedelta(days=step)

    return cutoffs
