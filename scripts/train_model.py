
"""
HORIZON ML - Filter Life Prediction
Model: Linear Regression
Validation: GroupShuffleSplit

Dataset is synthetic. Metrics describe only
performance on the simulated dataset.
"""

import json
import joblib
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "dataset" / "synthetic_filter_data.csv"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_FILE)

print("\n======================================")
print(" HORIZON ML - MODEL TRAINING")
print("======================================")

# Feature engineering
df["tds_difference"] = (
    df["inlet_tds"] - df["outlet_tds"]
)

df["tds_ratio"] = (
    df["outlet_tds"] /
    df["inlet_tds"].replace(0, np.nan)
)

df["usage_per_day"] = (
    df["cumulative_litres"] /
    df["age_days"].replace(0, np.nan)
)

# Replace undefined values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Initial observations have age_days = 0.
# Use daily_litres as their usage estimate.
df["usage_per_day"] = df["usage_per_day"].fillna(
    df["daily_litres"]
)

df["tds_ratio"] = df["tds_ratio"].fillna(0)

# Model input features
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

groups = df["cartridge_id"]

# Group-based train/test split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

train_groups = set(groups.iloc[train_indices])
test_groups = set(groups.iloc[test_indices])

# Verify no cartridge overlap
assert train_groups.isdisjoint(test_groups)

print("\n[1] DATA SPLIT")

print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))
print("Training cartridges:", len(train_groups))
print("Testing cartridges:", len(test_groups))
print("Cartridge overlap: NONE")

# Build model pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

# Train
print("\n[2] TRAINING MODEL")

model.fit(X_train, y_train)

print("Training completed.")

# Predictions
predictions = model.predict(X_test)

# Remaining life cannot be negative
predictions = np.maximum(predictions, 0)

# Evaluation
mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

# Additional percentage error
mape = np.mean(
    np.abs(
        (y_test.to_numpy() - predictions) /
        np.maximum(y_test.to_numpy(), 1)
    )
) * 100

print("\n[3] MODEL EVALUATION")

print(f"MAE:  {mae:.2f} days")
print(f"RMSE: {rmse:.2f} days")
print(f"R2:   {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

# Save model
model_path = MODEL_DIR / "horizon_filter_model.pkl"

joblib.dump(model, model_path)

# Save evaluation metrics
metrics = {
    "model": "Linear Regression",
    "dataset": "SYNTHETIC",
    "training_observations": len(X_train),
    "testing_observations": len(X_test),
    "training_cartridges": len(train_groups),
    "testing_cartridges": len(test_groups),
    "MAE_days": round(float(mae), 4),
    "RMSE_days": round(float(rmse), 4),
    "R2": round(float(r2), 4),
    "MAPE_percent": round(float(mape), 4)
}

metrics_path = RESULT_DIR / "model_metrics.json"

with open(metrics_path, "w") as file:
    json.dump(metrics, file, indent=4)

# Save predictions for analysis
comparison = pd.DataFrame({
    "actual_remaining_days": y_test.to_numpy(),
    "predicted_remaining_days": predictions,
    "absolute_error": np.abs(
        y_test.to_numpy() - predictions
    )
})

comparison.to_csv(
    RESULT_DIR / "prediction_comparison.csv",
    index=False
)

# Final output
print("\n[4] MODEL SAVED")

print("Model:", model_path)
print("Metrics:", metrics_path)
print("Predictions:", RESULT_DIR / "prediction_comparison.csv")

print("\n======================================")
print(" HORIZON MODEL TRAINING COMPLETE")
print("======================================")
