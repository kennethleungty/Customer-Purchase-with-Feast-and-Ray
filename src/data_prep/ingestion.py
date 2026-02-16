"""
Raw data ingestion and cleaning.
"""

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
