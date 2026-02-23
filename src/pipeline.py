"""
Data preparation pipeline orchestrator.
Uses Ray to parallelize feature engineering across rolling cutoff dates.
Each cutoff is an independent task, so Ray distributes them across available CPUs.
"""

import pandas as pd
import ray

from src.config import (
    BEHAVIOR_FEATURES_PATH,
    PURCHASE_WINDOW_DAYS,
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
    # Initialize Ray
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    # 1. Ingest and clean raw data
    df = ingest_and_clean(RAW_DATA_PATH)

    # 2. Generate rolling cutoff dates
    cutoffs = generate_cutoff_dates(
        df, FEATURE_WINDOW_DAYS, PURCHASE_WINDOW_DAYS, ROLLING_STEP_DAYS
    )

    # 3. Distribute feature engineering across Ray workers
    # ray.put() places DataFrame in shared object store once, avoiding redundant copies
    df_ref = ray.put(df)

    # Launch all cutoffs in parallel as Ray tasks
    futures = [
        compute_features_for_cutoff.remote(df_ref, cutoff, FEATURE_WINDOW_DAYS)
        for cutoff in cutoffs
    ]

    # Collect results from all workers
    results = ray.get(futures)

    # Extract and concatenate RFM and behavior features
    all_rfm = [r["rfm"] for r in results]
    all_behavior = [r["behavior"] for r in results]
    rfm_combined = pd.concat(all_rfm, ignore_index=True)
    behavior_combined = pd.concat(all_behavior, ignore_index=True)

    # Save to parquet files
    FEATURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    rfm_combined.to_parquet(RFM_FEATURES_PATH, index=False)
    behavior_combined.to_parquet(BEHAVIOR_FEATURES_PATH, index=False)

    ray.shutdown()


if __name__ == "__main__":
    run_data_prep_pipeline()
