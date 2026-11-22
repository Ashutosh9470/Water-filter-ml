
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent

RESULT_DIR = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "models"

df = pd.read_csv(
    BASE_DIR / "dataset" / "synthetic_filter_data.csv"
)

comparison = pd.read_csv(
    RESULT_DIR / "prediction_comparison.csv"
)

metrics = json.load(
    open(RESULT_DIR / "model_metrics.json")
)

model = joblib.load(
    MODEL_DIR / "horizon_filter_model.pkl"
)

# Actual and predicted values
actual = comparison["actual_remaining_days"]
predicted = comparison["predicted_remaining_days"]

errors = predicted - actual

print("\n================================")
print(" HORIZON MODEL DIAGNOSTICS")
print("================================")

# 1. Baseline comparison
baseline_prediction = np.repeat(
    metrics["training_observations"] * 0 +
    df["remaining_days"].mean(),
    len(actual)
)

# Use training-set mean reconstructed from
# the saved train/test cartridge split.
# A constant test-set mean is NOT a valid baseline.
# Report the test-set mean only as context.
print("\n[1] TEST SET ERROR")
print("Model MAE:", round(mean_absolute_error(actual, predicted), 2))
print("Model R2:", round(r2_score(actual, predicted), 4))

print(
    "Test target mean:",
    round(actual.mean(), 2)
)

# 2. Error by remaining-life range
print("\n[2] ERROR BY FILTER LIFE RANGE")

bins = [0, 30, 60, 90, 120, 180, 250, 1000]

labels = [
    "0-30",
    "30-60",
    "60-90",
    "90-120",
    "120-180",
    "180-250",
    "250+"
]

analysis = pd.DataFrame({
    "actual": actual,
    "predicted": predicted
})

analysis["life_range"] = pd.cut(
    analysis["actual"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

for label, group in analysis.groupby(
    "life_range", observed=False
):
    if len(group) > 0:
        mae = mean_absolute_error(
            group["actual"],
            group["predicted"]
        )

        print(
            f"{label} days: "
            f"MAE = {mae:.2f} days "
            f"(n={len(group)})"
        )

# 3. Bias
print("\n[3] PREDICTION BIAS")

print("Mean error:", round(errors.mean(), 2))
print("Positive = overprediction")
print("Negative = underprediction")

# 4. Coefficients
print("\n[4] MODEL COEFFICIENTS")

regressor = model.named_steps["regressor"]
scaler = model.named_steps["scaler"]

features = [
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

coefficients = pd.DataFrame({
    "feature": features,
    "coefficient": regressor.coef_
})

coefficients["absolute_effect"] = (
    coefficients["coefficient"].abs()
)

coefficients = coefficients.sort_values(
    "absolute_effect",
    ascending=False
)

print(coefficients.to_string(index=False))

coefficients.to_csv(
    RESULT_DIR / "model_coefficients.csv",
    index=False
)

# 5. Prediction scatter plot
plt.figure(figsize=(8, 6))

plt.scatter(
    actual,
    predicted,
    alpha=0.5
)

minimum = min(actual.min(), predicted.min())
maximum = max(actual.max(), predicted.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Remaining Life (days)")
plt.ylabel("Predicted Remaining Life (days)")
plt.title("Horizon: Actual vs Predicted Filter Life")

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "prediction_scatter.png",
    dpi=150
)

plt.show()

# 6. Residual plot
plt.figure(figsize=(8, 6))

plt.scatter(
    predicted,
    errors,
    alpha=0.5
)

plt.axhline(0, linestyle="--")

plt.xlabel("Predicted Remaining Life (days)")
plt.ylabel("Prediction Error (days)")
plt.title("Horizon: Residual Analysis")

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "residual_plot.png",
    dpi=150
)

plt.show()

print("\nDiagnostics saved in results/")
print("HORIZON DIAGNOSTICS COMPLETE")

