
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import GroupKFold
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

df = pd.read_csv(
    BASE_DIR / "dataset" / "synthetic_filter_data.csv"
)

RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

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
groups = df["cartridge_id"]

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

cv = GroupKFold(n_splits=5)

fold_results = []
all_predictions = []

print("\n======================================")
print(" HORIZON - 5 FOLD GROUP VALIDATION")
print("======================================")

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups), start=1
):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])

    assert train_groups.isdisjoint(test_groups)

    print(f"\nFOLD {fold}")
    print("Train cartridges:", len(train_groups))
    print("Test cartridges:", len(test_groups))

    # Training mean baseline
    baseline = np.full(
        len(y_test), y_train.mean()
    )

    predictions_by_model = {
        "Training Mean Baseline": baseline
    }

    for name, model_template in models.items():

        import copy
        model = copy.deepcopy(model_template)

        model.fit(X_train, y_train)

        predictions = np.maximum(
            model.predict(X_test), 0
        )

        predictions_by_model[name] = predictions

    for name, predictions in predictions_by_model.items():

        mae = mean_absolute_error(
            y_test, predictions
        )

        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )

        r2 = r2_score(y_test, predictions)

        fold_results.append({
            "fold": fold,
            "model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        all_predictions.append(
            pd.DataFrame({
                "fold": fold,
                "model": name,
                "actual": y_test.to_numpy(),
                "predicted": predictions
            })
        )

        print(
            f"{name}: MAE={mae:.2f}, "
            f"RMSE={rmse:.2f}, R2={r2:.4f}"
        )

# Combine results
fold_df = pd.DataFrame(fold_results)

prediction_df = pd.concat(
    all_predictions, ignore_index=True
)

# Mean and standard deviation across folds
summary = fold_df.groupby("model").agg(
    MAE_mean=("MAE", "mean"),
    MAE_std=("MAE", "std"),
    RMSE_mean=("RMSE", "mean"),
    RMSE_std=("RMSE", "std"),
    R2_mean=("R2", "mean"),
    R2_std=("R2", "std")
).reset_index()

# Pooled out-of-fold metrics
pooled = prediction_df.groupby("model").apply(
    lambda g: pd.Series({
        "pooled_MAE": mean_absolute_error(
            g["actual"], g["predicted"]
        ),
        "pooled_RMSE": np.sqrt(
            mean_squared_error(
                g["actual"], g["predicted"]
            )
        ),
        "pooled_R2": r2_score(
            g["actual"], g["predicted"]
        )
    }),
    include_groups=False
).reset_index()

summary = summary.merge(
    pooled, on="model"
)

print("\n======================================")
print(" FIVE FOLD SUMMARY")
print("======================================")

print(summary.round(4).to_string(index=False))

summary.to_csv(
    RESULT_DIR / "cross_validation_summary.csv",
    index=False
)

fold_df.to_csv(
    RESULT_DIR / "cross_validation_folds.csv",
    index=False
)

prediction_df.to_csv(
    RESULT_DIR / "cross_validation_predictions.csv",
    index=False
)

print("\nSaved cross-validation results in results/")
print("CROSS VALIDATION COMPLETE")
