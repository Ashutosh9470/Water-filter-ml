
"""
HORIZON ML - Filter Life Prediction Engine

Model: Random Forest Regressor
Input: Water quality + cartridge usage
Output: Estimated remaining filter life in days

IMPORTANT: Synthetic-data demonstration model.
"""

import joblib
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = BASE_DIR / "models" / "horizon_final_model.pkl"

# Check model availability
if not MODEL_FILE.exists():
    raise FileNotFoundError(
        "Final model not found. Run train_final_model.py first."
    )

# Load trained model
model = joblib.load(MODEL_FILE)


def get_float(prompt, minimum, maximum):

    while True:
        try:
            value = float(input(prompt))

            if not minimum <= value <= maximum:
                print(
                    f"Enter a value between {minimum} and {maximum}."
                )
                continue

            return value

        except ValueError:
            print("Invalid input. Enter a numeric value.")


def predict_filter_life():

    print("\n======================================")
    print("          HORIZON ML ENGINE")
    print("       SMART WATER PURIFIER")
    print("======================================")

    print("\nEnter cartridge information:\n")

    # Cartridge information
    age_days = get_float(
        "Cartridge age (days): ", 0, 3650
    )

    cumulative_litres = get_float(
        "Total water processed (litres): ", 0, 100000
    )

    daily_litres = get_float(
        "Average daily usage (litres): ", 0.01, 500
    )

    # Water quality readings
    print("\nEnter water quality readings:\n")

    inlet_tds = get_float(
        "Inlet TDS (ppm): ", 1, 2000
    )

    outlet_tds = get_float(
        "Outlet TDS (ppm): ", 0, 2000
    )

    tank_tds = get_float(
        "Tank TDS (ppm): ", 0, 2000
    )

    temperature = get_float(
        "Water temperature (C): ", 0, 60
    )

    if outlet_tds > inlet_tds:
        print(
            "\nWARNING: Outlet TDS exceeds inlet TDS. "
            "Verify the readings and sensor calibration."
        )

    # Feature engineering
    tds_difference = inlet_tds - outlet_tds

    tds_rejection_pct = (
        tds_difference / inlet_tds
    ) * 100

    tds_ratio = outlet_tds / inlet_tds

    usage_per_day = (
        cumulative_litres / age_days
        if age_days > 0
        else daily_litres
    )

    # Features must match training order exactly
    input_data = pd.DataFrame([{
        "age_days": age_days,
        "cumulative_litres": cumulative_litres,
        "daily_litres": daily_litres,
        "inlet_tds": inlet_tds,
        "outlet_tds": outlet_tds,
        "tank_tds": tank_tds,
        "temperature": temperature,
        "tds_rejection_pct": tds_rejection_pct,
        "tds_difference": tds_difference,
        "tds_ratio": tds_ratio,
        "usage_per_day": usage_per_day
    }])

    # Predict
    prediction = model.predict(input_data)[0]

    prediction = max(0, prediction)

    # Display results
    print("\n======================================")
    print("          PREDICTION RESULT")
    print("======================================")

    print(
        f"\nEstimated remaining life: "
        f"{prediction:.1f} DAYS"
    )

    print(
        f"Estimated remaining life: "
        f"{prediction / 30.44:.1f} MONTHS"
    )

    print("\nWater quality information:")
    print(f"Inlet TDS: {inlet_tds:.1f} ppm")
    print(f"Outlet TDS: {outlet_tds:.1f} ppm")
    print(f"TDS rejection: {tds_rejection_pct:.2f}%")

    print("\nModel information:")
    print("Algorithm: Random Forest")
    print("Data basis: SYNTHETIC")
    print("Status: DEMONSTRATION MODEL")

    print("\nNOTE:")
    print(
        "This is an experimental estimate from a "
        "synthetic-data model, not a validated "
        "real-world cartridge-life measurement."
    )

    print("\n======================================")


if __name__ == "__main__":
    predict_filter_life()
