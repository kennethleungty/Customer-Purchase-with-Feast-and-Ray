.PHONY: db prep apply train predict all clean

# ── Full pipeline (run all steps in order) ────────────────────────────────────
all: db prep apply train

# ── Infrastructure ────────────────────────────────────────────────────────────

# Start PostgreSQL (Feast registry backend) via Docker
db:
	docker compose up -d
	@echo "Waiting for PostgreSQL to be ready..."
	@sleep 3

# ── Individual steps ──────────────────────────────────────────────────────────

# Step 1: Ingest raw data, engineer features, save parquets
prep:
	python -m src.pipeline

# Step 2: Register Feast feature definitions in PostgreSQL registry
apply:
	cd feature_store && feast apply

# Step 3: Train XGBoost using features retrieved from Feast
train:
	python -m src.train

# Step 4: Batch-score all customers using Feast + saved model
predict:
	python -m src.predict

# ── Utilities ─────────────────────────────────────────────────────────────────

# Remove all generated artifacts (parquets, model, predictions)
clean:
	rm -rf feature_store/data/ models/

# Stop and remove PostgreSQL container + volume
clean-db:
	docker compose down -v
