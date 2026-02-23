# Setup and Run Guide: Feast & Ray for Feature Engineering

This guide walks through the complete setup and execution flow for using **Feast** (feature store) and **Ray** (distributed computing) for feature engineering in this customer purchase propensity project.

---

## Overview

This project demonstrates two key uses of Feast and Ray:

1. **Ray for Distributed Feature Engineering** (`pipeline.py`)
   - Parallelizes feature computation across rolling cutoff dates
   - Each cutoff date runs as an independent `@ray.remote` task
   - All cutoffs execute simultaneously instead of sequentially

2. **Feast for Feature Management & Retrieval** (`train.py`, `predict.py`)
   - Centralized feature definitions with lineage and schemas
   - Point-in-time correct feature retrieval (prevents data leakage)
   - Ray-backed offline store for distributed parquet reads and joins

---

## Step 1: Initial Setup (Dependencies & Infrastructure)

### 1.1 Install Python Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Key dependencies installed:**
- `feast[ray]>=0.40.0` — Feature store with Ray offline store support
- `ray>=2.53.0` — Distributed compute framework
- `psycopg2-binary>=2.9.11` — PostgreSQL adapter for Feast registry
- `pandas`, `xgboost`, `scikit-learn` — Data processing and ML

### 1.2 Prepare Raw Data

Place the UCI Online Retail dataset (`Online Retail.xlsx`) in:
```
data/input/Online Retail.xlsx
```

### 1.3 Start PostgreSQL (Feast Registry Backend)

```bash
make db
# or: docker compose up -d
```

**What this does:**
- Starts PostgreSQL 16 container (`feast-postgres`) on port 5433
- Creates the `feast_registry` database
- Credentials: `feast:feast` (user:password)

**Why PostgreSQL instead of local SQLite?**
- Multi-user access: multiple data scientists can read/write to the same registry
- Production-ready: simulates a real feature store infrastructure
- Concurrent access: supports team collaboration

**Registry purpose:**
- Stores feature metadata: entity schemas, feature view definitions, data source locations
- Lightweight catalog (small, write-heavy during development)
- Separate from actual feature data (which lives in parquet files)

---

## Step 2: Data Preparation & Ray-Based Feature Engineering

```bash
make prep
# or: python -m src.pipeline
```

### What Happens Here

**2.1 Data Ingestion** (`src/data_prep/ingestion.py`)
- Loads `Online Retail.xlsx`
- Cleans data: removes rows without CustomerID, calculates Revenue, flags cancellations
- Output: ~400K transactions, ~3,800 customers

**2.2 Generate Rolling Cutoff Dates** (`src/data_prep/cutoffs.py`)
- Creates ~9 cutoff dates spaced 30 days apart
- Each cutoff defines a temporal snapshot:
  - **Features**: computed from 90-day window BEFORE cutoff `[C - 90d, C)`
  - **Labels**: computed from 30-day window AFTER cutoff `[C, C + 30d)`
- Ensures enough historical data for features and future data for labels

**2.3 Distributed Feature Engineering with Ray** (`src/pipeline.py`)

This is where Ray parallelizes feature computation:

```python
@ray.remote
def compute_features_for_cutoff(df, cutoff, feature_window):
    """Ray remote task: compute features for a single cutoff"""
    rfm = build_rfm_features(df, cutoff, feature_window)
    behavior = build_behavior_features(df, cutoff, feature_window)
    return {"cutoff": cutoff, "rfm": rfm, "behavior": behavior}
```

**Ray execution flow:**
1. **Initialize Ray**: `ray.init()` — uses all available CPUs by default
2. **Share data once**: `df_ref = ray.put(df)` — places DataFrame in Ray's object store, avoiding redundant copies to each worker
3. **Launch parallel tasks**: Create one `@ray.remote` task per cutoff date
4. **Collect results**: `ray.get(futures)` — blocks until all tasks complete
5. **Shutdown**: `ray.shutdown()`

**Why Ray here?**
- **Sequential problem**: Computing features for 9 cutoffs sequentially takes 9× the time
- **Embarrassingly parallel**: Each cutoff is independent — no cross-dependencies
- **Ray solution**: Distributes cutoffs across CPU cores, reducing total time to ~1× (plus overhead)

**Features Computed:**

**RFM Features** (`src/feature_engineering/rfm_features.py`):
- `recency_days`: days since last purchase in 90-day window
- `frequency`: number of distinct orders in 90-day window
- `monetary`: total spend in 90-day window
- `tenure_days`: days since customer's first-ever purchase (all-time)

**Behavioral Features** (`src/feature_engineering/behavior_features.py`):
- `avg_order_value`: mean spend per order
- `avg_basket_size`: mean items per order
- `n_unique_products`: product diversity
- `return_rate`: share of cancelled orders
- `avg_days_between_purchases`: purchase cadence

**Outputs:**
- `feature_store/data/customer_rfm_features.parquet` (~17K rows)
- `feature_store/data/customer_behavior_features.parquet` (~17K rows)

Each parquet contains:
- Multi-snapshot data: same customer appears at multiple `event_timestamp` values
- Columns: `customer_id`, `event_timestamp`, feature columns
- One row per `(customer_id, cutoff_date)` combination

---

## Step 3: Register Feast Feature Definitions

```bash
make apply
# or: cd feature_store && feast apply
```

### What Happens Here

Feast reads `feature_store/definitions.py` and registers the following in PostgreSQL:

**3.1 Entity Definition**
```python
customer = Entity(
    name="customer",
    join_keys=["customer_id"],
    value_type=ValueType.INT64,
)
```
- The "primary key" that ties feature rows to real-world objects
- All feature views in this project use `customer_id` as the join key

**3.2 Data Sources**
```python
rfm_source = FileSource(
    path="feature_store/data/customer_rfm_features.parquet",
    timestamp_field="event_timestamp",
)
behavior_source = FileSource(
    path="feature_store/data/customer_behavior_features.parquet",
    timestamp_field="event_timestamp",
)
```
- Points Feast to the parquet files created in Step 2
- `timestamp_field` is CRITICAL for point-in-time joins

**3.3 Feature Views**
```python
customer_rfm_fv = FeatureView(
    name="customer_rfm_features",
    entities=[customer],
    ttl=timedelta(days=0),  # features never expire
    schema=[
        Field(name="recency_days", dtype=Int64),
        Field(name="frequency", dtype=Int64),
        # ... more fields
    ],
    source=rfm_source,
)
```

**Why 2 feature views instead of 1?**
- In production, different feature groups often have:
  - Different refresh cadences (e.g., RFM daily, behavior weekly)
  - Different owners/teams
  - Different data sources
- Splitting them demonstrates Feast's ability to join multiple feature views at retrieval time

**Registry after `feast apply`:**
- PostgreSQL `feast_registry` database now contains:
  - 1 entity: `customer`
  - 2 data sources: RFM and Behavior parquets
  - 2 feature views: `customer_rfm_features`, `customer_behavior_features`
  - Schemas, types, metadata for all features

---

## Step 4: Setup Entity DataFrame (for Training)

This step is part of `train.py` but is conceptually important:

### 4.1 Build Entity DataFrame

```python
def build_entity_df() -> pd.DataFrame:
    """
    Create entity DataFrame with ALL rolling cutoff dates.
    Each row = (customer_id, event_timestamp, purchased)
    """
    # Load raw data and generate cutoffs
    df = ingest_and_clean(RAW_DATA_PATH)
    cutoffs = generate_cutoff_dates(df, ...)
    
    # Compute purchase labels for each cutoff
    all_labels = []
    for cutoff in cutoffs:
        labels = build_purchase_labels(df, cutoff, ...)
        labels["event_timestamp"] = cutoff
        all_labels.append(labels)
    
    return pd.concat(all_labels)
```

**Entity DataFrame structure:**
```
customer_id | event_timestamp       | purchased
------------|----------------------|----------
12345       | 2010-12-01 00:00:00 | 1
12345       | 2010-12-31 00:00:00 | 0
12345       | 2011-01-30 00:00:00 | 1
...         | ...                  | ...
```

**Key insight:**
- Same customer appears at multiple `event_timestamp` values
- `event_timestamp` is the cutoff date, NOT the transaction date
- This is the "spine" that Feast will join features onto

**Purchase labels:**
- `purchased = 1`: customer made at least one purchase in [cutoff, cutoff + 30 days)
- `purchased = 0`: customer made zero purchases in that window
- Computed on-the-fly during training from raw data

---

## Step 5: Retrieve Features from Feast (Point-in-Time Join)

```bash
make train
# or: python -m src.train
```

### 5.1 Feast Feature Retrieval

```python
from feast import FeatureStore

store = FeatureStore(repo_path="feature_store/")

# Specify which features to retrieve
feature_refs = [
    "customer_rfm_features:recency_days",
    "customer_rfm_features:frequency",
    "customer_rfm_features:monetary",
    "customer_rfm_features:tenure_days",
    "customer_behavior_features:avg_order_value",
    # ... more features
]

# Point-in-time join: Feast retrieves correct feature snapshot for each row
training_df = store.get_historical_features(
    entity_df=entity_df,  # from Step 4
    features=feature_refs,
).to_df()
```

### What Happens Under the Hood

**5.2 Feast Configuration** (`feature_store/feature_store.yaml`):
```yaml
project: retail_purchase
registry:
    registry_type: sql
    path: postgresql+psycopg2://feast:feast@localhost:5433/feast_registry
offline_store:
    type: ray  # Uses Ray for distributed joins
    storage_path: data/ray_storage
```

**5.3 Ray Offline Store in Action**

When `get_historical_features()` is called:

1. **Registry lookup**: Feast queries PostgreSQL to find:
   - Where the parquet files are located
   - What columns exist in each feature view
   - Which timestamp field to use

2. **Ray-distributed read**: Instead of loading parquets in a single process (pandas):
   - Ray offline store spawns multiple workers
   - Each worker reads a partition of the parquet files
   - Parallelizes the I/O across CPU cores

3. **Point-in-time join**: For EACH feature view:
   - Feast joins `entity_df` with the feature view's data source
   - **Critical rule**: For each `(customer_id, event_timestamp)` in entity_df, retrieve the feature row where:
     - `customer_id` matches
     - Feature's `event_timestamp` <= entity's `event_timestamp`
     - Use the feature row with the LATEST timestamp that satisfies the above
   - This ensures NO data leakage: features at time T only use data available at or before T

4. **Combine feature views**: Feast joins RFM and Behavior features separately, then merges them

5. **Return DataFrame**: Final training_df has:
   - All entity_df columns (`customer_id`, `event_timestamp`, `purchased`)
   - All requested features from both feature views

**Why Ray offline store matters:**
- With 2 feature views, that's 2 temporal joins
- Entity DataFrame has ~17K rows (3,800 customers × 9 cutoffs)
- As datasets grow to millions of rows and more feature views are added, these joins become the bottleneck
- Ray parallelizes this work, keeping retrieval times manageable at scale

---

## Step 6: Train Model Using Retrieved Features

Continuing in `train.py`:

### 6.1 Temporal Train/Test Split

```python
# CRITICAL: temporal split, not random split
last_cutoff = training_df["event_timestamp"].max()
train_mask = training_df["event_timestamp"] < last_cutoff
test_mask = training_df["event_timestamp"] == last_cutoff

X_train = training_df.loc[train_mask, ALL_FEATURES]
y_train = training_df.loc[train_mask, "purchased"]
X_test = training_df.loc[test_mask, ALL_FEATURES]
y_test = training_df.loc[test_mask, "purchased"]
```

**Why temporal split?**
- Random split would leak future information into training set
- Temporal split simulates real-world scenario: train on past, predict future
- Test set = last cutoff date = genuinely future data

### 6.2 Train XGBoost

```python
model = XGBClassifier(**XGB_PARAMS)
model.fit(X_train, y_train)
model.save_model("models/xgb_purchase_model.json")
```

**Model outputs:**
- Accuracy, F1 score, ROC-AUC on test set
- Saved model: `models/xgb_purchase_model.json` (XGBoost native JSON format)

---

## Step 7: Batch Prediction Using Feast

```bash
make predict
# or: python -m src.predict
```

### 7.1 Build Entity DataFrame (Latest Cutoff Only)

```python
# Read RFM parquet to get all available snapshots
rfm_df = pd.read_parquet(
    "feature_store/data/customer_rfm_features.parquet",
    columns=["customer_id", "event_timestamp"]
)

# Filter to LATEST cutoff only
latest_cutoff = rfm_df["event_timestamp"].max()
entity_df = rfm_df[rfm_df["event_timestamp"] == latest_cutoff]
```

**Key difference from training:**
- Training entity_df: ALL cutoff dates (~17K rows)
- Prediction entity_df: ONLY latest cutoff (~3,800 rows)
- Same customers, but only their most recent feature snapshot

### 7.2 Retrieve Features from Feast (Same API)

```python
store = FeatureStore(repo_path="feature_store/")
features_df = store.get_historical_features(
    entity_df=entity_df,  # latest cutoff only
    features=feature_refs,  # same features as training
).to_df()
```

**Feast ensures consistency:**
- Same feature definitions used in training and prediction
- Same retrieval logic (point-in-time join)
- No training-serving skew

### 7.3 Generate Predictions

```python
model = XGBClassifier()
model.load_model("models/xgb_purchase_model.json")

X = features_df[ALL_FEATURES].fillna(0)
predictions = features_df[["customer_id"]].copy()
predictions["purchase_probability"] = model.predict_proba(X)[:, 1]
predictions["purchase_predicted"] = model.predict(X)

predictions.to_parquet("models/predictions.parquet")
```

**Outputs:**
- `models/predictions.parquet` with columns:
  - `customer_id`
  - `purchase_probability` (0-1 score)
  - `purchase_predicted` (binary 0/1)

---

## Summary: Complete Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Initial Setup                                       │
│  - Install dependencies (feast[ray], ray, pandas, etc.)    │
│  - Start PostgreSQL (Feast registry backend)               │
│  - Prepare raw data (Online Retail.xlsx)                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Ray-Based Feature Engineering (pipeline.py)        │
│  - Generate rolling cutoff dates (~9 cutoffs)              │
│  - Ray parallelizes feature computation across cutoffs     │
│  - Each cutoff = independent @ray.remote task               │
│  - Output: 2 multi-snapshot parquet files                  │
│    • customer_rfm_features.parquet (~17K rows)             │
│    • customer_behavior_features.parquet (~17K rows)        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Register Feast Feature Definitions (feast apply)   │
│  - Read feature_store/definitions.py                       │
│  - Register in PostgreSQL registry:                        │
│    • 1 Entity (customer)                                   │
│    • 2 FileSource (RFM, Behavior parquets)                 │
│    • 2 FeatureView (customer_rfm, customer_behavior)       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Setup Entity DataFrame (train.py)                  │
│  - Build entity_df with ALL cutoff dates                   │
│  - Compute purchase labels on-the-fly                      │
│  - Structure: (customer_id, event_timestamp, purchased)    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Retrieve Features from Feast (train.py)            │
│  - store.get_historical_features(entity_df, features)      │
│  - Ray offline store: distributed parquet reads + joins    │
│  - Point-in-time join: prevents data leakage               │
│  - Output: training_df with all features + labels          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 6: Train Model (train.py)                             │
│  - Temporal train/test split (last cutoff = test)          │
│  - Train XGBoost classifier                                 │
│  - Save model: xgb_purchase_model.json                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 7: Batch Prediction (predict.py)                      │
│  - Entity DataFrame: LATEST cutoff only                    │
│  - Retrieve features from Feast (same API)                 │
│  - Load trained model                                       │
│  - Generate predictions: purchase probability + binary     │
│  - Save: predictions.parquet                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Concepts Recap

### Ray for Feature Engineering
- **Problem**: Sequential feature computation is slow (9 cutoffs = 9× time)
- **Solution**: `@ray.remote` tasks parallelize across CPU cores
- **Where**: `src/pipeline.py` — distributed feature engineering
- **Result**: ~1× time (plus overhead) instead of 9× time

### Feast for Feature Management
- **Problem**: Feature inconsistency, poor lineage tracking, data leakage risks
- **Solution**: Centralized feature definitions + point-in-time joins
- **Where**: `train.py`, `predict.py` — feature retrieval
- **Components**:
  - **Registry** (PostgreSQL): metadata catalog (schemas, definitions)
  - **Offline store** (Ray-backed): actual feature data + distributed joins
  - **Entity DataFrame**: "spine" that defines what to retrieve
  - **Point-in-time join**: ensures features at time T only use data ≤ T

### Why This Architecture?
1. **Ray in pipeline.py**: Speeds up feature computation (embarrassingly parallel problem)
2. **Feast in train.py/predict.py**: Ensures feature consistency and prevents data leakage
3. **Two separate uses**: Ray for computation (pipeline), Feast+Ray for retrieval (training/prediction)
4. **Production-ready**: PostgreSQL registry, Ray offline store, Docker infra

---

## Production Next Steps

### Scale Ray
- Deploy Ray cluster instead of local single-node
- Use `ray_address` config in Feast to point to remote cluster
- Or use KubeRay for Kubernetes-based elastic scaling

### Scale Feast
- **Registry**: Replace local PostgreSQL with managed database (Cloud SQL, RDS)
- **Offline store**: Swap `FileSource` for `BigQuerySource`/`SnowflakeSource`
- **Online store**: Add Redis for real-time serving (sub-10ms latency)

### CI/CD
- Run `feast apply` in CI pipeline
- Version control feature definitions
- Automated tests for feature schemas

---

## Command Reference

```bash
# Full pipeline
make all          # db → prep → apply → train

# Individual steps
make db           # Start PostgreSQL (Feast registry)
make prep         # Ray-based feature engineering → parquets
make apply        # Register Feast definitions in PostgreSQL
make train        # Feast retrieval → train XGBoost
make predict      # Batch prediction via Feast

# Cleanup
make clean        # Remove parquets, model, predictions
make clean-db     # Stop PostgreSQL container + remove volume
```
