
import json
import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "dataset" / "synthetic_filter_data.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)

# Feature engineering
df["tds_difference"] = (
    df["inlet_tds"] - df["outlet_tds"]
)

df["tds_ratio"] = (
    df["outlet_tds"] /
    df["inlet_tds"].replace(0, np.nan)
).fillna(0)

df["usage_per_day"] = (
    df["cumulative_litres"] /
    df["age_days"].replace(0, np.nan)
).fillna(df["daily_litres"])

FEATURES = [
    "age_days",
    "cumulative_litres",
    "daily_litres",
    "inlet_tds",
    "outlet_tds",
    "tank_tds",
    "temperature",
    "tds_rejection_pct",
    "tds_difference",
    "tds_ratio",
    "usage_per_day"
]

TARGET = "remaining_days"

X = df[FEATURES]
y = df[TARGET]

print("\n================================")
print(" HORIZON FINAL MODEL TRAINING")
print("================================")

print("Dataset observations:", len(df))
print("Cartridges:", df["cartridge_id"].nunique())

# Final Random Forest
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

# Save model
model_path = MODEL_DIR / "horizon_final_model.pkl"

joblib.dump(model, model_path)

# Save metadata
metadata = {
    "project": "HORIZON",
    "model": "Random Forest Regressor",
    "model_type": "SYNTHETIC_PROTOTYPE",
    "dataset": "synthetic_filter_data.csv",
    "observations": len(df),
    "cartridges": int(df["cartridge_id"].nunique()),
    "features": FEATURES,
    "target": TARGET,
    "estimators": 300,
    "max_depth": 12,
    "random_state": 42,
    "validation": "5-fold GroupKFold by cartridge",
    "pooled_cv_mae_days": 30.6316,
    "pooled_cv_rmse_days": 40.9466,
    "pooled_cv_r2": 0.8379
}

metadata_path = MODEL_DIR / "model_metadata.json"

with open(metadata_path, "w") as file:
    json.dump(metadata, file, indent=4)

print("\nFINAL MODEL TRAINED")
print("Model:", model_path)
print("Metadata:", metadata_path)

print("\nModel configuration:")
print("Algorithm: Random Forest")
print("Trees: 300")
print("Maximum depth: 12")
print("Training observations:", len(X))

print("\n================================")
print(" HORIZON FINAL MODEL READY")
print("================================")
