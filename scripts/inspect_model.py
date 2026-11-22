
import joblib
import json
import numpy as np

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = BASE_DIR / "models" / "horizon_final_model.pkl"
METADATA_FILE = BASE_DIR / "models" / "model_metadata.json"

model = joblib.load(MODEL_FILE)

with open(METADATA_FILE) as f:
    metadata = json.load(f)

print("\n================================")
print(" HORIZON MODEL INSPECTION")
print("================================")

print("Algorithm:", metadata["model"])
print("Trees:", len(model.estimators_))
print("Features:", len(metadata["features"]))

total_nodes = 0
maximum_depth = 0

for tree in model.estimators_:
    estimator = tree.tree_

    total_nodes += estimator.node_count

    maximum_depth = max(
        maximum_depth,
        estimator.max_depth
    )

print("Total decision tree nodes:", total_nodes)
print("Maximum tree depth:", maximum_depth)

print("\nFEATURE IMPORTANCE")

importance = model.feature_importances_

for feature, value in sorted(
    zip(metadata["features"], importance),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{feature:25s}: {value:.5f}")

print("\nMODEL FILE SIZE")

size_bytes = MODEL_FILE.stat().st_size

print(f"Size: {size_bytes:,} bytes")
print(f"Size: {size_bytes / 1024:.2f} KB")

print("\n================================")
print(" INSPECTION COMPLETE")
print("================================")
