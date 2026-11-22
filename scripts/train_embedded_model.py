
import json
import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "dataset" / "synthetic_filter_data.csv"
MODEL_DIR = BASE_DIR / "models"
FIRMWARE_DIR = BASE_DIR / "firmware"

MODEL_DIR.mkdir(exist_ok=True)
FIRMWARE_DIR.mkdir(exist_ok=True)

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

X = df[FEATURES]
y = df["remaining_days"]

print("\n================================")
print(" HORIZON EMBEDDED MODEL")
print("================================")

# Train complete-data embedded candidate
model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

model.fit(X, y)

scaler = model.named_steps["scaler"]
regressor = model.named_steps["regressor"]

means = scaler.mean_
scales = scaler.scale_
coefficients = regressor.coef_
intercept = regressor.intercept_

# Save Python model
joblib.dump(
    model,
    MODEL_DIR / "horizon_embedded_model.pkl"
)

# Export model parameters
parameters = {
    "model": "Linear Regression",
    "data_source": "SYNTHETIC",
    "features": FEATURES,
    "means": means.tolist(),
    "scales": scales.tolist(),
    "coefficients": coefficients.tolist(),
    "intercept": float(intercept)
}

json_path = FIRMWARE_DIR / "model_parameters.json"

with open(json_path, "w") as f:
    json.dump(parameters, f, indent=4)

# Generate C++ header
header = [
    "#pragma once",
    "",
    "// HORIZON ML embedded model",
    "// Trained using synthetic demonstration data",
    "",
    "constexpr int FEATURE_COUNT = 11;",
    "",
    "const float FEATURE_MEANS[FEATURE_COUNT] = {"
]

header.append(
    "    " + ", ".join(f"{x:.10f}f" for x in means)
)

header.extend([
    "};",
    "",
    "const float FEATURE_SCALES[FEATURE_COUNT] = {"
])

header.append(
    "    " + ", ".join(f"{x:.10f}f" for x in scales)
)

header.extend([
    "};",
    "",
    "const float MODEL_COEFFICIENTS[FEATURE_COUNT] = {"
])

header.append(
    "    " + ", ".join(
        f"{x:.10f}f" for x in coefficients
    )
)

header.extend([
    "};",
    "",
    f"const float MODEL_INTERCEPT = {intercept:.10f}f;",
    "",
])

header_path = FIRMWARE_DIR / "horizon_model.h"

header_path.write_text("\n".join(header))

# Verify exported mathematical model against sklearn
scaled_X = (X.to_numpy() - means) / scales

manual_predictions = (
    scaled_X @ coefficients + intercept
)

sklearn_predictions = model.predict(X)

maximum_difference = np.max(
    np.abs(manual_predictions - sklearn_predictions)
)

print("\nMODEL TRAINED")
print("Training observations:", len(X))
print("Training cartridges:", df["cartridge_id"].nunique())

print("\nEXPORT FILES")
print("Python model:", MODEL_DIR / "horizon_embedded_model.pkl")
print("JSON parameters:", json_path)
print("C++ header:", header_path)

print("\nEXPORT VERIFICATION")
print("Maximum prediction difference:", maximum_difference)

if maximum_difference < 0.0001:
    print("STATUS: PASSED")
else:
    print("STATUS: FAILED")

print("\n================================")
print(" EMBEDDED MODEL EXPORT COMPLETE")
print("================================")
