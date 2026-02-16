"""
Batch prediction pipeline
=========================
1. Build an entity DataFrame for the LATEST cutoff date only
2. Retrieve features from Feast offline store
3. Load the trained XGBoost model
4. Generate purchase predictions + probabilities
5. Save results to parquet
"""

import pandas as pd
from feast import FeatureStore
from xgboost import XGBClassifier

from src.config import (
    ALL_FEATURES,
    BEHAVIOR_FEATURES,
    FEATURE_STORE_DIR,
    MODEL_DIR,
    MODEL_PATH,
    PREDICTIONS_PATH,
    RFM_FEATURES,
    RFM_FEATURES_PATH,
)


def main():
    print("=== Batch Prediction Pipeline ===\n")

    # 1. Entity DataFrame — score customers at the latest cutoff only.
    #    The parquet contains rows at many cutoff dates (rolling window).
    #    We filter to the most recent one for "current" predictions.
    rfm_df = pd.read_parquet(
        RFM_FEATURES_PATH, columns=["customer_id", "event_timestamp"]
    )
    latest_cutoff = rfm_df["event_timestamp"].max()
    entity_df = rfm_df[rfm_df["event_timestamp"] == latest_cutoff][
        ["customer_id", "event_timestamp"]
    ].copy()
    print(f"[1/4] {len(entity_df)} customers to score (cutoff: {latest_cutoff.date()})")

    # 2. Retrieve features from Feast (same retrieval logic as training)
    print("[2/4] Retrieving features from Feast...")
    store = FeatureStore(repo_path=str(FEATURE_STORE_DIR))

    # Build feature refs from config lists — single source of truth
    feature_refs = [f"customer_rfm_features:{f}" for f in RFM_FEATURES] + [
        f"customer_behavior_features:{f}" for f in BEHAVIOR_FEATURES
    ]

    features_df = store.get_historical_features(
        entity_df=entity_df,
        features=feature_refs,
    ).to_df()

    # 3. Load trained model
    print("[3/4] Loading model...")
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))

    # 4. Predict
    X = features_df[ALL_FEATURES].fillna(0)
    predictions = features_df[["customer_id"]].copy()
    predictions["purchase_probability"] = model.predict_proba(X)[:, 1]
    predictions["purchase_predicted"] = model.predict(X)

    # 5. Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(str(PREDICTIONS_PATH), index=False)

    n_purchasers = predictions["purchase_predicted"].sum()
    print("\n[4/4] Results:")
    print(f"      Total customers:       {len(predictions)}")
    print(f"      Predicted purchasers:  {n_purchasers}")
    print(f"      Saved to {PREDICTIONS_PATH}\n")


if __name__ == "__main__":
    main()
