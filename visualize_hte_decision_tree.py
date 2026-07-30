from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from decision_tree_pie_viz import DecisionTreePieVizConfig, save_decision_tree_pies


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "OER_HTS" / "input" / "paper" / "results" / "96_conditions_results.csv"
OUTPUT_IMAGE = BASE_DIR / "OER_HTS" / "output" / "decision_tree_pies.png"
TARGET_COLUMN = "overpotential(mV)"
DEFAULT_FONT_SCALE = 1.3
FEATURE_COLUMNS = [
    "Ni/Fe",
    "KNO3",
    "potential(V)",
    "time(s)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and visualize the HTE decision tree.")
    parser.add_argument(
        "--font-scale",
        type=float,
        default=DEFAULT_FONT_SCALE,
        help="Global font-size multiplier for the saved tree figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))

    df = pd.read_csv(INPUT_CSV)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    model = DecisionTreeRegressor(random_state=42, max_depth=3)
    model.fit(X, y)

    config = DecisionTreePieVizConfig(
        pie_bins=4,
        binning="quantile",
        transparent_background=True,
        font_scale=args.font_scale,
        min_node_radius_px=18.0,
        max_node_radius_px=84.0,
    )

    _, _, output_path = save_decision_tree_pies(
        model,
        X,
        y,
        OUTPUT_IMAGE,
        target_name=TARGET_COLUMN,
        config=config,
    )

    print(f"Loaded rows: {len(df)}")
    print(f"Rows used for training: {len(X)}")
    print(f"Feature columns: {FEATURE_COLUMNS}")
    print(f"Target column: {TARGET_COLUMN}")
    print(f"Font scale: {args.font_scale}")
    print(f"Saved custom tree plot to: {output_path}")


if __name__ == "__main__":
    main()
