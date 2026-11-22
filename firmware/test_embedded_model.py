
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

model = joblib.load(ROOT / "models" / "horizon_embedded_model.pkl")

with open(ROOT / "firmware" / "model_parameters.json") as f:
    params = json.load(f)

# Sample sensor readings
age_days = 60
cumulative_litres = 2400
daily_litres = 40
inlet_tds = 550
outlet_tds = 65
tank_tds = 60
temperature = 27

tds_rejection_pct = ((inlet_tds - outlet_tds) / inlet_tds) * 100
tds_difference = inlet_tds - outlet_tds
tds_ratio = outlet_tds / inlet_tds
usage_per_day = cumulative_litres / age_days

features = np.array([[
    age_days,
    cumulative_litres,
    daily_litres,
    inlet_tds,
    outlet_tds,
    tank_tds,
    temperature,
    tds_rejection_pct,
    tds_difference,
    tds_ratio,
    usage_per_day
]])

# Python prediction
feature_df = pd.DataFrame(features, columns=params["features"])
python_prediction = max(0, model.predict(feature_df)[0])

# Embedded C++ equivalent
means = np.array(params["means"])
scales = np.array(params["scales"])
coefficients = np.array(params["coefficients"])
intercept = params["intercept"]

standardized = (features[0] - means) / scales

embedded_prediction = max(
    0,
    intercept + np.dot(standardized, coefficients)
)

difference = abs(python_prediction - embedded_prediction)

print("\n================================")
print(" HORIZON EMBEDDED MODEL TEST")
print("================================")
print(f"Python prediction:   {python_prediction:.4f} days")
print(f"Embedded prediction: {embedded_prediction:.4f} days")
print(f"Difference:          {difference:.10f}")
print("--------------------------------")

if difference < 0.0001:
    print("STATUS: PASSED")
else:
    print("STATUS: FAILED")

print("================================")
