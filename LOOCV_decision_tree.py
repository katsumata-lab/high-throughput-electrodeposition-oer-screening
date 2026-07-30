from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeRegressor

d = None

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "input" / "results" / "96_conditions_results.csv"
OUTPUT_CSV = BASE_DIR / "output" / f"decision_tree_loocv_predictions_{d}.csv"
TARGET_COLUMN = "overpotential(mV)"
FEATURE_COLUMNS = [
    "Ni/Fe",
    "KNO3",
    "potential(V)",
    "time(s)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run leave-one-out cross-validation with the HTE decision tree and save "
            "predictions alongside descriptor values."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_CSV,
        help=f"Input CSV path (default: {INPUT_CSV})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help=f"Output CSV path (default: {OUTPUT_CSV})",
    )
    return parser.parse_args()


def load_valid_dataset(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    selected = df[required_columns].copy()
    valid_mask = selected[FEATURE_COLUMNS].notna().all(axis=1) & selected[TARGET_COLUMN].notna()
    selected = selected.loc[valid_mask].reset_index(names="source_row_index")

    if len(selected) < 2:
        raise ValueError("At least two valid rows are required to run LOOCV.")

    return selected


def run_loocv(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    loo = LeaveOneOut()
    prediction_rows: list[dict[str, float | int]] = []

    for train_index, test_index in loo.split(X):
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        test_row = df.iloc[test_index[0]]

        model = DecisionTreeRegressor(random_state=42, max_depth=d)
        model.fit(X_train, y_train)
        predicted_value = float(model.predict(X_test)[0])
        actual_value = float(test_row[TARGET_COLUMN])

        row = {"source_row_index": int(test_row["source_row_index"])}
        for column in FEATURE_COLUMNS:
            row[column] = test_row[column]
        row["actual_value"] = actual_value
        row["predicted_value"] = predicted_value
        row["residual"] = predicted_value - actual_value
        row["absolute_error"] = abs(predicted_value - actual_value)
        prediction_rows.append(row)

    return pd.DataFrame(prediction_rows)


def main() -> None:
    args = parse_args()

    valid_df = load_valid_dataset(args.input)
    results_df = run_loocv(valid_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False)

    mae = mean_absolute_error(results_df["actual_value"], results_df["predicted_value"])
    rmse = math.sqrt(
        mean_squared_error(
            results_df["actual_value"],
            results_df["predicted_value"],
        )
    )
    r2 = r2_score(results_df["actual_value"], results_df["predicted_value"])

    print(f"Loaded valid rows: {len(valid_df)}")
    print(f"Feature columns: {FEATURE_COLUMNS}")
    print(f"Target column: {TARGET_COLUMN}")
    print(f"LOOCV folds: {len(results_df)}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")
    print(f"Saved LOOCV predictions to: {args.output}")


if __name__ == "__main__":
    main()
