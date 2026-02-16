"""
Data preparation pipeline orchestrator.
Uses Ray to parallelize feature engineering across rolling cutoff dates.
Each cutoff is an independent task, so Ray distributes them across available CPUs.
"""

import pandas as pd
import ray

from src.config import (
    BEHAVIOR_FEATURES_PATH,
    CHURN_WINDOW_DAYS,
    FEATURE_DATA_DIR,
    FEATURE_WINDOW_DAYS,
    RAW_DATA_PATH,
    RFM_FEATURES_PATH,
    ROLLING_STEP_DAYS,
)
from src.data_prep import generate_cutoff_dates, ingest_and_clean
from src.feature_engineering import build_behavior_features, build_rfm_features


@ray.remote
def compute_features_for_cutoff(
    df: pd.DataFrame, cutoff: pd.Timestamp, feature_window: int
) -> dict:
    """
    Ray remote task: compute RFM and behavioral features for a single cutoff.
    Runs in a separate worker process, enabling parallel execution across cutoffs.
    Returns a dict with the two DataFrames so Ray can serialize them back.
    """
    rfm = build_rfm_features(df, cutoff, feature_window)
    behavior = build_behavior_features(df, cutoff, feature_window)

    # Tag with event_timestamp so Feast can do point-in-time joins
    rfm["event_timestamp"] = cutoff
    behavior["event_timestamp"] = cutoff

    rfm = rfm.rename(columns={"CustomerID": "customer_id"})
    behavior = behavior.rename(columns={"CustomerID": "customer_id"})

    return {"cutoff": cutoff, "rfm": rfm, "behavior": behavior}


def run_data_prep_pipeline():
    print("=== Data Preparation Pipeline ===\n")

    # Initialize Ray (uses all available CPUs by default)
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    print(f"      Ray initialized — {ray.cluster_resources().get('CPU', 0):.0f} CPUs available\n")

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

    # 3. Fan out feature engineering across Ray workers.
    #    ray.put() places the DataFrame in shared object store once,
    #    avoiding redundant copies to each worker.
    print(f"\n[3/3] Engineering features at each cutoff (window={FEATURE_WINDOW_DAYS}d)...")
    df_ref = ray.put(df)

    # Launch all cutoffs in parallel as Ray tasks
    futures = [
        compute_features_for_cutoff.remote(df_ref, cutoff, FEATURE_WINDOW_DAYS)
        for cutoff in cutoffs
    ]

    # Collect results as they complete
    results = ray.get(futures)

    # Sort by cutoff date for deterministic output order
    results.sort(key=lambda r: r["cutoff"])

    all_rfm = []
    all_behavior = []
    for r in results:
        all_rfm.append(r["rfm"])
        all_behavior.append(r["behavior"])
        print(f"      {r['cutoff'].date()}: {len(r['rfm'])} customers")

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

    ray.shutdown()


if __name__ == "__main__":
    run_data_prep_pipeline()
