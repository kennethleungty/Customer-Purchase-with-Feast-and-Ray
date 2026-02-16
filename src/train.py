"""
Training pipeline
=================
1. Compute churn labels at runtime for all rolling cutoff dates
2. Retrieve features from Feast's offline store (point-in-time join)
3. Temporal train / test split (earlier cutoffs = train, last cutoff = test)
4. Train an XGBoost binary classifier
5. Evaluate (accuracy, F1, ROC-AUC) and save the model
"""

import pandas as pd
from feast import FeatureStore
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.config import (
    ALL_FEATURES,
    BEHAVIOR_FEATURES,
    CHURN_WINDOW_DAYS,
    FEATURE_STORE_DIR,
    FEATURE_WINDOW_DAYS,
    MODEL_DIR,
    MODEL_PATH,
    RAW_DATA_PATH,
    RFM_FEATURES,
    ROLLING_STEP_DAYS,
    XGB_PARAMS,
)
from src.data_prep import (
    build_churn_labels,
    generate_cutoff_dates,
    ingest_and_clean,
)


def build_entity_df() -> pd.DataFrame:
    """
    Build the entity DataFrame for ALL rolling cutoff dates.

    For each cutoff, computes churn labels on the fly from raw data.
    The same customer can appear at multiple cutoffs with different labels
    (e.g., retained at cutoff_1 but churned at cutoff_5).

    Each row = (customer_id, event_timestamp, churn).
    Feast will use (customer_id, event_timestamp) to fetch the correct
    feature snapshot via point-in-time join.
    """
    df = ingest_and_clean(RAW_DATA_PATH)
    cutoffs = generate_cutoff_dates(
        df, FEATURE_WINDOW_DAYS, CHURN_WINDOW_DAYS, ROLLING_STEP_DAYS
    )

    all_labels = []
    for cutoff in cutoffs:
        labels = build_churn_labels(df, cutoff, CHURN_WINDOW_DAYS, FEATURE_WINDOW_DAYS)
        labels = labels.rename(columns={"CustomerID": "customer_id"})
        labels["event_timestamp"] = cutoff
        all_labels.append(labels)

    return pd.concat(all_labels, ignore_index=True)


def retrieve_features(entity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Use Feast to join features from BOTH feature views onto the entity df.

    With rolling windows, the entity df contains rows at many different
    event_timestamps.  Feast's point-in-time join matches each
    (customer_id, event_timestamp) to the feature row whose timestamp
    is <= the requested timestamp — this is the core mechanism that
    prevents data leakage across time.
    """
    store = FeatureStore(repo_path=str(FEATURE_STORE_DIR))

    # Build feature refs from config lists — add features in config.py
    # and they'll automatically be requested here.
    feature_refs = [
        f"customer_rfm_features:{f}" for f in RFM_FEATURES
    ] + [
        f"customer_behavior_features:{f}" for f in BEHAVIOR_FEATURES
    ]

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=feature_refs,
    ).to_df()

    return training_df


def main():
    print("=== Training Pipeline ===\n")

    # 1. Entity DataFrame with churn labels across all cutoffs
    print("[1/5] Building entity DataFrame (all cutoff dates)...")
    entity_df = build_entity_df()
    n_cutoffs = entity_df["event_timestamp"].nunique()
    print(f"      {len(entity_df):,} rows across {n_cutoffs} cutoffs")

    # 2. Retrieve features from Feast offline store
    print("[2/5] Retrieving features from Feast (point-in-time join)...")
    training_df = retrieve_features(entity_df)
    print(f"      {training_df.shape[1]} columns × {len(training_df):,} rows")

    # 3. Temporal split: train on all cutoffs except the last,
    #    test on the last cutoff (genuinely future data).
    last_cutoff = training_df["event_timestamp"].max()
    train_mask = training_df["event_timestamp"] < last_cutoff
    test_mask = training_df["event_timestamp"] == last_cutoff

    X_train = training_df.loc[train_mask, ALL_FEATURES].fillna(0)
    y_train = training_df.loc[train_mask, "churn"]
    X_test = training_df.loc[test_mask, ALL_FEATURES].fillna(0)
    y_test = training_df.loc[test_mask, "churn"]

    print(f"\n[3/5] Temporal train/test split (test = {last_cutoff.date()}):")
    print(f"      Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    print(f"      Train churn rate: {y_train.mean():.1%}")
    print(f"      Test  churn rate: {y_test.mean():.1%}")

    # 4. Train XGBoost
    print("\n[4/5] Training XGBoost classifier...")
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # 5. Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n[5/5] Evaluation metrics:")
    print(f"      Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"      F1 Score : {f1_score(y_test, y_pred):.4f}")
    print(f"      ROC AUC  : {roc_auc_score(y_test, y_prob):.4f}")
    print(
        f"\n{classification_report(y_test, y_pred, target_names=['Retained', 'Churned'])}"
    )

    # 6. Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    print(f"      Model saved to {MODEL_PATH}\n")


if __name__ == "__main__":
    main()
