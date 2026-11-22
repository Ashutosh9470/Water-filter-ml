
import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# HORIZON - SENSOR DATA GENERATOR
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
DATASET_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = DATASET_DIR / "synthetic_sensor_data.csv"

RATED_CAPACITY_LITRES = 9463.53
NUMBER_OF_CARTRIDGES = 50
DAYS_PER_CARTRIDGE = 365

rng = np.random.default_rng(42)

all_records = []

for cartridge_number in range(1, NUMBER_OF_CARTRIDGES + 1):

    cartridge_id = f"HZ_{cartridge_number:03d}"

    # Simulated cartridge characteristics
    base_daily_usage = rng.uniform(15, 40)
    base_turbidity = rng.uniform(0.3, 1.5)

    cumulative_litres = 0.0
    daily_readings = []

    for day in range(1, DAYS_PER_CARTRIDGE + 1):

        # Daily water consumption
        daily_litres = max(
            0,
            rng.normal(base_daily_usage, 4)
        )

        cumulative_litres += daily_litres

        # Flow sensor: instantaneous flow rate
        flow_rate = max(
            0,
            rng.normal(1.8, 0.35)
        )

        # Temperature sensor
        temperature = rng.normal(27, 3)

        # Synthetic outlet turbidity.
        # This is an illustrative simulation, not measured
        # physical behaviour of the actual cartridge.
        usage_fraction = min(
            cumulative_litres / RATED_CAPACITY_LITRES,
            1.5
        )

        turbidity = (
            base_turbidity
            + 0.8 * usage_fraction
            + rng.normal(0, 0.12)
        )

        turbidity = max(0, turbidity)

        daily_readings.append(turbidity)

        # Historical turbidity features
        turbidity_7d_avg = np.mean(
            daily_readings[-7:]
        )

        if len(daily_readings) >= 7:
            turbidity_7d_previous = np.mean(
                daily_readings[-14:-7]
            ) if len(daily_readings) >= 14 else daily_readings[0]

            turbidity_trend = (
                turbidity_7d_avg - turbidity_7d_previous
            )
        else:
            turbidity_trend = 0.0

        # Capacity-based remaining life
        remaining_litres = max(
            0,
            RATED_CAPACITY_LITRES - cumulative_litres
        )

        remaining_life_pct = max(
            0,
            min(
                100,
                (remaining_litres / RATED_CAPACITY_LITRES) * 100
            )
        )

        remaining_days = (
            remaining_litres / base_daily_usage
            if base_daily_usage > 0
            else 0
        )

        daily_records = {
            "cartridge_id": cartridge_id,
            "day": day,
            "age_days": day,
            "daily_litres": round(daily_litres, 3),
            "cumulative_litres": round(cumulative_litres, 3),
            "flow_rate": round(flow_rate, 3),
            "temperature": round(temperature, 3),
            "outlet_turbidity": round(turbidity, 3),
            "turbidity_7d_avg": round(turbidity_7d_avg, 3),
            "turbidity_trend": round(turbidity_trend, 4),
            "remaining_litres": round(remaining_litres, 3),
            "remaining_life_pct": round(remaining_life_pct, 3),
            "remaining_days": round(remaining_days, 2),
            "data_source": "SYNTHETIC"
        }

        all_records.append(daily_records)

# Create DataFrame
df = pd.DataFrame(all_records)

# Save dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\n================================")
print(" HORIZON SENSOR DATA GENERATOR")
print("================================")

print(f"Cartridges generated: {NUMBER_OF_CARTRIDGES}")
print(f"Total observations: {len(df)}")
print(f"Rated capacity: {RATED_CAPACITY_LITRES} L")

print("\nDataset columns:")
print(list(df.columns))

print("\nSample records:")
print(df.head(5).to_string(index=False))

print("\nDataset statistics:")
print(df[[
    "daily_litres",
    "flow_rate",
    "temperature",
    "outlet_turbidity",
    "remaining_life_pct"
]].describe().round(2))

print("\nData source:", df["data_source"].unique())

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nSTATUS: DATA GENERATION COMPLETE")
print("================================")
