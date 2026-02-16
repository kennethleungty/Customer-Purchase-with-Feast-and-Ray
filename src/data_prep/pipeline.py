"""
Data preparation pipeline orchestrator.
Loops over rolling cutoff dates, engineers features at each, and saves parquets.
"""

import pandas as pd

from src.config import (
    BEHAVIOR_FEATURES_PATH,
    CHURN_WINDOW_DAYS,
    FEATURE_DATA_DIR,
    FEATURE_WINDOW_DAYS,
    RAW_DATA_PATH,
    RFM_FEATURES_PATH,
    ROLLING_STEP_DAYS,
)
from src.data_prep.behavior_features import build_behavior_features
from src.data_prep.rfm_features import build_rfm_features
from src.data_prep.utils import generate_cutoff_dates, ingest_and_clean


def main():
    print("=== Data Preparation Pipeline ===\n")

    # 1. Ingest and clean
    print("[1/3] Loading raw data...")
    df = ingest_and_clean(RAW_DATA_PATH)
    print(f"      {len(df):,} rows | {df['CustomerID'].nunique():,} customers\n")

    # 2. Generate rolling cutoff dates
    cutoffs = generate_cutoff_dates(
        df, FEATURE_WINDOW_DAYS, CHURN_WINDOW_DAYS, ROLLING_STEP_DAYS
    )
    print(f"[2/3] Generated {len(cutoffs)} rolling cutoff dates:")
    for c in cutoffs:
        print(f"      {c.date()}")

    # 3. At each cutoff, compute features from the 90-day window before it.
    #    Each customer can appear at multiple cutoffs (if they were active
    #    in that window), giving us many more training samples.
    print(f"\n[3/3] Engineering features at each cutoff (window={FEATURE_WINDOW_DAYS}d)...")
    all_rfm = []
    all_behavior = []

    for cutoff in cutoffs:
        rfm = build_rfm_features(df, cutoff, FEATURE_WINDOW_DAYS)
        behavior = build_behavior_features(df, cutoff, FEATURE_WINDOW_DAYS)

        # Tag with event_timestamp so Feast can do point-in-time joins
        rfm["event_timestamp"] = cutoff
        behavior["event_timestamp"] = cutoff

        rfm = rfm.rename(columns={"CustomerID": "customer_id"})
        behavior = behavior.rename(columns={"CustomerID": "customer_id"})

        all_rfm.append(rfm)
        all_behavior.append(behavior)

        print(f"      {cutoff.date()}: {len(rfm)} customers")

    # Concatenate all snapshots into single parquets
    rfm_combined = pd.concat(all_rfm, ignore_index=True)
    behavior_combined = pd.concat(all_behavior, ignore_index=True)

    total_rows = len(rfm_combined)
    unique_customers = rfm_combined["customer_id"].nunique()
    print(f"\n      Total: {total_rows:,} rows ({unique_customers:,} unique customers)")

    # Save
    FEATURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    rfm_combined.to_parquet(RFM_FEATURES_PATH, index=False)
    behavior_combined.to_parquet(BEHAVIOR_FEATURES_PATH, index=False)

    print(f"      Parquets saved to {FEATURE_DATA_DIR}/")
    print("      Done!\n")


if __name__ == "__main__":
    main()
