from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))

from matplotlib import colors as mcolors
import matplotlib.pyplot as plt


DEFAULT_RESULTS_DIR = BASE_DIR / "input" / "results"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output" / "results"

# Edit these values when running this script from an IDE.
INPUT_CSV = DEFAULT_RESULTS_DIR / "96_conditions_results.csv"
OUTPUT_PATH = None
# Columns used to create the heatmap.  Change these names to match the CSV header.
X_AXIS_COLUMN = "potential(V)"
Y_AXIS_COLUMN = "time(s)"
COLOR_COLUMN = "overpotential(V)"
STD_COLUMN = "standard_deviation"
FIXED_CONDITIONS = {
    "Ni/Fe": 1,
    "KNO3": 0,
}
OUTPUT_DPI = 300
CMAP = "GnBu"
COLORBAR_MIN = 250
COLORBAR_MAX = 500
FONT_SIZE_SCALE = 2.0
ANNOTATE_VALUES = True
ANNOTATE_STD = False


class HeatmapColumns(NamedTuple):
    conditions: list[str]
    x_axis: str
    y_axis: str
    value: str
    std: str


def parse_fixed_condition(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            "Fixed conditions must be written as COLUMN=VALUE, "
            'for example: --fix "ECD_time(s)=60"'
        )
    column, value = text.split("=", 1)
    column = column.strip()
    value = value.strip()
    if not column or not value:
        raise argparse.ArgumentTypeError("Both COLUMN and VALUE are required in --fix.")
    return column, value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a heatmap from an HTE result CSV. Axis and color columns are "
            "configured at the top of this script."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        nargs="?",
        default=INPUT_CSV,
        help=(
            "Input CSV path, or a filename under input/HTE/paper/results "
            f"(default: {INPUT_CSV})"
        ),
    )
    parser.add_argument(
        "--fix",
        dest="fixed_conditions",
        type=parse_fixed_condition,
        action="append",
        default=None,
        help=(
            "Condition value to fix, written as COLUMN=VALUE. Specify exactly two. "
            'Example: --fix "potential(V)=-1.2" --fix "time(s)=60"'
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(OUTPUT_PATH) if OUTPUT_PATH else None,
        help="Output image path. Default is generated under output/HTE/paper/plots.",
    )
    parser.add_argument("--dpi", type=int, default=OUTPUT_DPI)
    parser.add_argument("--cmap", default=CMAP)
    parser.add_argument(
        "--vmin",
        type=float,
        default=COLORBAR_MIN,
        help="Minimum value for the heatmap color scale.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=COLORBAR_MAX,
        help="Maximum value for the heatmap color scale.",
    )
    parser.set_defaults(annotate=ANNOTATE_VALUES)
    parser.add_argument("--annotate", dest="annotate", action="store_true")
    parser.add_argument("--no-annotate", dest="annotate", action="store_false")
    parser.set_defaults(annotate_std=ANNOTATE_STD)
    parser.add_argument(
        "--annotate-std",
        dest="annotate_std",
        action="store_true",
        help="Show standard deviation in each heatmap cell as a second line.",
    )
    args = parser.parse_args()
    if args.fixed_conditions is None:
        args.fixed_conditions = list(FIXED_CONDITIONS.items())
    return args


def get_config() -> argparse.Namespace:
    if len(sys.argv) == 1:
        return argparse.Namespace(
            input_csv=Path(INPUT_CSV),
            fixed_conditions=list(FIXED_CONDITIONS.items()),
            output=Path(OUTPUT_PATH) if OUTPUT_PATH else None,
            dpi=OUTPUT_DPI,
            cmap=CMAP,
            vmin=COLORBAR_MIN,
            vmax=COLORBAR_MAX,
            annotate=ANNOTATE_VALUES,
            annotate_std=ANNOTATE_STD,
        )
    return parse_args()


def resolve_input_path(path_or_name: Path) -> Path:
    if path_or_name.is_absolute() or path_or_name.parent != Path("."):
        path = path_or_name
    else:
        path = DEFAULT_RESULTS_DIR / path_or_name

    if path.exists():
        return path

    # The requested name is easy to mistype because the existing dummy file uses a hyphen.
    hyphen_fallback = path.with_name(path.name.replace("96_conditions", "96-conditions"))
    if hyphen_fallback.exists():
        return hyphen_fallback

    return path


def default_output_path_with_conditions(
    input_path: Path,
    fixed_conditions: dict[str, object],
) -> Path:
    fixed_suffix = fixed_conditions_filename_suffix(fixed_conditions)
    return DEFAULT_OUTPUT_DIR / f"{input_path.stem}_overpotential_heatmap_{fixed_suffix}.png"


def fixed_conditions_filename_suffix(fixed_conditions: dict[str, object]) -> str:
    parts = []
    for column, value in fixed_conditions.items():
        condition_name = column_without_unit(column)
        parts.append(f"{condition_name}-{format_value(value)}")
    return "__".join(safe_filename_part(part) for part in parts)


def column_without_unit(column: str) -> str:
    column = normalize_column_name(column)
    column = re.sub(r"\([^)]*\)", "", column)
    column = re.sub(r"\([^)]*$", "", column)
    column = column.strip("_- ")
    column = re.sub(r"_ratio$", "", column)
    return re.sub(r"\s+", " ", column).strip()


def column_match_key(column: str) -> str:
    column = column_without_unit(column).lower()
    return re.sub(r"[^a-z0-9]+", "", column)


def safe_filename_part(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    return text.strip("-") or "condition"


def normalize_column_name(column: object) -> str:
    return str(column).replace("\ufeff", "").strip()


def load_results(input_path: Path) -> tuple[pd.DataFrame, HeatmapColumns]:
    df = pd.read_csv(input_path)
    df.columns = [normalize_column_name(column) for column in df.columns]

    x_axis = resolve_csv_column(X_AXIS_COLUMN, list(df.columns))
    y_axis = resolve_csv_column(Y_AXIS_COLUMN, list(df.columns))
    value = resolve_csv_column(COLOR_COLUMN, list(df.columns))
    std = resolve_csv_column(STD_COLUMN, list(df.columns))
    if len({x_axis, y_axis, value, std}) != 4:
        raise ValueError(
            "X_AXIS_COLUMN, Y_AXIS_COLUMN, COLOR_COLUMN, and STD_COLUMN "
            "must each refer to a different CSV column."
        )

    conditions = [column for column in df.columns if column not in {value, std}]
    if x_axis not in conditions or y_axis not in conditions:
        raise ValueError("The X and Y axis columns must be condition columns.")
    columns = HeatmapColumns(
        conditions=conditions,
        x_axis=x_axis,
        y_axis=y_axis,
        value=value,
        std=std,
    )
    used_columns = columns.conditions + [columns.value, columns.std]
    df = df.loc[:, used_columns].copy()

    for column in used_columns:
        df[column] = normalize_values(df[column])

    df[columns.value] = pd.to_numeric(df[columns.value], errors="coerce")
    df[columns.std] = pd.to_numeric(df[columns.std], errors="coerce")
    return df, columns


def resolve_csv_column(column: str, available_columns: list[str]) -> str:
    if column in available_columns:
        return column

    requested = column_match_key(column)
    matches = [
        candidate
        for candidate in available_columns
        if column_match_key(candidate) == requested
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Configured column '{column}' is ambiguous: {', '.join(matches)}")
    raise ValueError(
        f"Configured column '{column}' was not found. "
        f"Available columns: {', '.join(available_columns)}"
    )


def normalize_values(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.map(lambda value: value.strip() if isinstance(value, str) else value)

    converted = pd.to_numeric(series, errors="coerce")
    if converted.notna().sum() == series.notna().sum():
        return converted
    return series


def make_fixed_condition_map(
    fixed_conditions: list[tuple[str, object]], condition_columns: list[str]
) -> dict[str, object]:
    fixed = {}
    for column, value in fixed_conditions:
        resolved_column = resolve_condition_column(column, condition_columns)
        if resolved_column in fixed:
            raise ValueError(f"Duplicate fixed condition column: {resolved_column}")
        fixed[resolved_column] = value

    if len(fixed) != 2:
        raise ValueError(
            "Specify exactly two fixed conditions. "
            f"Available condition columns: {', '.join(condition_columns)}"
        )

    return fixed


def resolve_condition_column(column: str, condition_columns: list[str]) -> str:
    if column in condition_columns:
        return column

    requested = column_match_key(column)
    matches = [
        candidate
        for candidate in condition_columns
        if column_match_key(candidate) == requested
    ]
    if not matches:
        matches = [
            candidate
            for candidate in condition_columns
            if column_match_key(candidate).startswith(requested)
        ]
    if not matches:
        matches = [
            candidate
            for candidate in condition_columns
            if requested in column_match_key(candidate)
        ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise ValueError(
            f"Fixed condition column '{column}' is ambiguous. "
            f"Matched: {', '.join(matches)}"
        )

    raise ValueError(
        f"Unknown fixed condition column: {column}. "
        f"Available condition columns: {', '.join(condition_columns)}"
    )


def matches_value(series: pd.Series, value: object) -> pd.Series:
    numeric_series = pd.to_numeric(series, errors="coerce")
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric_value):
        return np.isclose(numeric_series, float(numeric_value), rtol=0, atol=1e-9)

    return series.astype(str).str.strip() == str(value).strip()


def filter_rows(df: pd.DataFrame, fixed_conditions: dict[str, object]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for column, value in fixed_conditions.items():
        mask &= matches_value(df[column], value)
    return df.loc[mask].copy()


def build_heatmap_tables(
    df: pd.DataFrame,
    columns: HeatmapColumns,
    fixed_conditions: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, str, str, pd.DataFrame]:
    filtered = filter_rows(df, fixed_conditions)
    if filtered.empty:
        available = (
            df.loc[:, columns.conditions]
            .drop_duplicates()
            .sort_values(columns.conditions)
            .to_string(index=False)
        )
        raise ValueError(
            "No rows matched the fixed conditions.\n"
            f"Fixed conditions: {fixed_conditions}\n"
            f"Available condition combinations:\n{available}"
        )

    if columns.x_axis in fixed_conditions or columns.y_axis in fixed_conditions:
        raise ValueError(
            "X_AXIS_COLUMN and Y_AXIS_COLUMN cannot also be specified in FIXED_CONDITIONS."
        )

    x_column, y_column = columns.x_axis, columns.y_axis
    value_table = filtered.pivot_table(
        index=y_column,
        columns=x_column,
        values=columns.value,
        aggfunc="mean",
        dropna=False,
    )
    std_table = filtered.pivot_table(
        index=y_column,
        columns=x_column,
        values=columns.std,
        aggfunc="mean",
        dropna=False,
    )
    value_table = sort_table(value_table)
    std_table = std_table.reindex(index=value_table.index, columns=value_table.columns)
    return value_table, std_table, x_column, y_column, filtered


def sort_table(table: pd.DataFrame) -> pd.DataFrame:
    return table.sort_index(axis=0).sort_index(axis=1)


def format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):g}"
    return str(value)


def draw_heatmap(
    value_table: pd.DataFrame,
    std_table: pd.DataFrame,
    columns: HeatmapColumns,
    x_column: str,
    y_column: str,
    fixed_conditions: dict[str, object],
    output_path: Path,
    *,
    dpi: int,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    annotate: bool,
    annotate_std: bool,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))

    width = max(5.0, 1.1 * len(value_table.columns) + 2.8) * 1.25
    height = max(4.0, 0.8 * len(value_table.index) + 2.4) * 1.25
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)

    matrix = value_table.to_numpy(dtype=float)
    image = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    tick_font_size = 10 * FONT_SIZE_SCALE
    label_font_size = 10 * FONT_SIZE_SCALE

    ax.set_xticks(range(len(value_table.columns)))
    ax.set_xticklabels(
        [format_value(value) for value in value_table.columns],
        fontsize=tick_font_size,
    )
    ax.set_yticks(range(len(value_table.index)))
    ax.set_yticklabels(
        [format_value(value) for value in value_table.index],
        fontsize=tick_font_size,
    )
    ax.set_xlabel(x_column, fontsize=label_font_size)
    ax.set_ylabel(y_column, fontsize=label_font_size)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(columns.value, fontsize=label_font_size)
    colorbar.ax.tick_params(labelsize=tick_font_size)

    if annotate:
        add_cell_annotations(ax, value_table, std_table, annotate_std, cmap, image.norm)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def add_cell_annotations(
    ax: plt.Axes,
    value_table: pd.DataFrame,
    std_table: pd.DataFrame,
    annotate_std: bool,
    cmap_name: str,
    norm: mcolors.Normalize,
) -> None:
    colormap = plt.get_cmap(cmap_name)

    for row_index, row_label in enumerate(value_table.index):
        for column_index, column_label in enumerate(value_table.columns):
            value = value_table.loc[row_label, column_label]
            if pd.isna(value):
                continue

            label = format_value(value)
            if annotate_std:
                std_value = std_table.loc[row_label, column_label]
                if not pd.isna(std_value):
                    label = f"{label}\n+/- {format_value(std_value)}"

            red, green, blue, _ = colormap(norm(float(value)))
            luminance = 0.299 * red + 0.587 * green + 0.114 * blue
            text_color = "black" if luminance > 0.55 else "white"
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=9 * FONT_SIZE_SCALE,
            )


def main() -> None:
    config = get_config()
    if config.vmin is not None and config.vmax is not None and config.vmin >= config.vmax:
        raise ValueError(f"--vmin must be smaller than --vmax: {config.vmin} >= {config.vmax}")

    input_path = resolve_input_path(config.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df, columns = load_results(input_path)
    fixed_conditions = make_fixed_condition_map(config.fixed_conditions, columns.conditions)
    value_table, std_table, x_column, y_column, filtered = build_heatmap_tables(
        df,
        columns,
        fixed_conditions,
    )

    output_path = config.output or default_output_path_with_conditions(
        input_path,
        fixed_conditions,
    )
    draw_heatmap(
        value_table,
        std_table,
        columns,
        x_column,
        y_column,
        fixed_conditions,
        output_path,
        dpi=config.dpi,
        cmap=config.cmap,
        vmin=config.vmin,
        vmax=config.vmax,
        annotate=config.annotate,
        annotate_std=config.annotate_std,
    )

    print(f"Loaded rows: {len(df)}")
    print(f"Rows used for heatmap: {len(filtered)}")
    print(f"Fixed conditions: {fixed_conditions}")
    print(f"X axis: {x_column}")
    print(f"Y axis: {y_column}")
    print(f"Heatmap values: {columns.value}")
    print(f"Standard deviation column: {columns.std}")
    print(f"Color scale: vmin={config.vmin}, vmax={config.vmax}")
    print(f"Saved heatmap to: {output_path}")


if __name__ == "__main__":
    main()
