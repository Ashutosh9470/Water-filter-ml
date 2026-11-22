
import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "dataset" / "synthetic_filter_data.csv"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)

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

df["tds_ratio"] = df["tds_ratio"].fillna(0)

df["usage_per_day"] = df["usage_per_day"].fillna(
    df["daily_litres"]
)

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

# Same grouped split for all models
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])

assert train_groups.isdisjoint(test_groups)

print("\n======================================")
print(" HORIZON - MODEL COMPARISON")
print("======================================")

print("Training cartridges:", len(train_groups))
print("Testing cartridges:", len(test_groups))
print("Cartridge overlap: NONE")

# Define models
models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=3,
        loss="huber",
        random_state=42
    )
}

results = []
trained_models = {}

# Training mean baseline
baseline_prediction = np.full(
    len(y_test),
    y_train.mean()
)

baseline_mae = mean_absolute_error(
    y_test, baseline_prediction
)

baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_prediction)
)

baseline_r2 = r2_score(
    y_test, baseline_prediction
)

results.append({
    "Model": "Training Mean Baseline",
    "MAE": baseline_mae,
    "RMSE": baseline_rmse,
    "R2": baseline_r2
})

print("\nTraining mean baseline:")
print(f"MAE:  {baseline_mae:.2f}")
print(f"RMSE: {baseline_rmse:.2f}")
print(f"R2:   {baseline_r2:.4f}")

# Train and evaluate
for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Prevent negative remaining-life estimates
    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(
        y_test, predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(
        y_test, predictions
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[name] = model

    print(f"MAE:  {mae:.2f} days")
    print(f"RMSE: {rmse:.2f} days")
    print(f"R2:   {r2:.4f}")

# Comparison table
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "MAE"
).reset_index(drop=True)

print("\n======================================")
print(" FINAL COMPARISON")
print("======================================")

print(
    results_df.to_string(
        index=False,
        formatters={
            "MAE": "{:.2f}".format,
            "RMSE": "{:.2f}".format,
            "R2": "{:.4f}".format
        }
    )
)

# Save comparison
results_df.to_csv(
    RESULT_DIR / "model_comparison.csv",
    index=False
)

# Select best ML model by MAE
best_name = results_df[
    results_df["Model"] != "Training Mean Baseline"
].iloc[0]["Model"]

best_model = trained_models[best_name]

joblib.dump(
    best_model,
    MODEL_DIR / "horizon_best_model.pkl"
)

print("\nBest model by test MAE:", best_name)
print("Saved: models/horizon_best_model.pkl")
print("Comparison saved: results/model_comparison.csv")

print("\n======================================")
print(" MODEL COMPARISON COMPLETE")
print("======================================")
