
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FILE = BASE_DIR / "dataset" / "synthetic_sensor_data.csv"

df = pd.read_csv(FILE)

required_columns = [
    "cartridge_id",
    "day",
    "age_days",
    "daily_litres",
    "cumulative_litres",
    "flow_rate",
    "temperature",
    "outlet_turbidity",
    "turbidity_7d_avg",
    "turbidity_trend",
    "remaining_litres",
    "remaining_life_pct",
    "remaining_days",
    "data_source"
]

errors = []

print("\n================================")
print(" HORIZON DATA VALIDATION")
print("================================")

# 1. Required columns
missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    errors.append(f"Missing columns: {missing_columns}")

# 2. Missing values
missing_values = df.isnull().sum().sum()

if missing_values:
    errors.append(f"Missing values: {missing_values}")

# 3. Duplicate observations
duplicates = df.duplicated(
    subset=["cartridge_id", "day"]
).sum()

if duplicates:
    errors.append(f"Duplicate observations: {duplicates}")

# 4. Cartridge count
cartridges = df["cartridge_id"].nunique()

if cartridges != 50:
    errors.append(f"Expected 50 cartridges, found {cartridges}")

# 5. Data source
if not (df["data_source"] == "SYNTHETIC").all():
    errors.append("Unexpected data source values")

# 6. Sensor range checks
range_checks = {
    "daily_litres": (0, 100),
    "flow_rate": (0, 20),
    "temperature": (0, 100),
    "outlet_turbidity": (0, 100),
    "remaining_life_pct": (0, 100),
    "remaining_litres": (0, 9463.53)
}

for column, (low, high) in range_checks.items():
    invalid = (
        (df[column] < low) |
        (df[column] > high)
    ).sum()

    if invalid:
        errors.append(
            f"{column}: {invalid} out-of-range values"
        )

# 7. Check cumulative usage
for cartridge_id, group in df.groupby("cartridge_id"):
    group = group.sort_values("day")

    if not group["cumulative_litres"].is_monotonic_increasing:
        errors.append(
            f"Cumulative usage decreases: {cartridge_id}"
        )

    if not group["age_days"].is_monotonic_increasing:
        errors.append(
            f"Age timeline invalid: {cartridge_id}"
        )

# 8. Verify capacity calculation
expected_remaining = np.maximum(
    0,
    9463.53 - df["cumulative_litres"]
)

capacity_error = np.abs(
    expected_remaining - df["remaining_litres"]
).max()

if capacity_error > 0.01:
    errors.append("Remaining litres calculation incorrect")

expected_percentage = (
    expected_remaining / 9463.53
) * 100

percentage_error = np.abs(
    expected_percentage - df["remaining_life_pct"]
).max()

if percentage_error > 0.01:
    errors.append("Remaining life percentage incorrect")

# 9. Report
print(f"Total observations: {len(df)}")
print(f"Unique cartridges: {cartridges}")
print(f"Missing values: {missing_values}")
print(f"Duplicate observations: {duplicates}")
print(f"Data source: {df['data_source'].unique()}")

print("\nCapacity calculation:")
print(f"Maximum remaining-litres error: {capacity_error:.6f}")
print(f"Maximum percentage error: {percentage_error:.6f}")

print("\nValidation results:")

if errors:
    for error in errors:
        print("[FAIL]", error)

    print("\nSTATUS: FAILED")
else:
    print("All validation checks passed.")
    print("\nSTATUS: PASSED")

print("================================")
