from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# =======================================================
# User settings
# =======================================================
INPUT_CSV = Path("output/OER/run_YYYYMMDD_HHMMSS/combined_current_density.csv")
OUTPUT_CSV = INPUT_CSV.with_name("overpotential_at_target_current.csv")

TARGET_CURRENT_DENSITY_MA_CM2 = 1.0
THEORETICAL_POTENTIAL_V_VS_RHE = 1.23

# Number of consecutive points that must be at or above the target current
# density before the threshold is accepted. This suppresses isolated noise.
MIN_CONSECUTIVE_POINTS = 3

# Potential changes smaller than this value are treated as no scan movement.
POTENTIAL_STEP_TOLERANCE_V = 1.0e-9

TIME_COLUMN = "time_s"
POTENTIAL_COLUMN = "potential_V_vs_RHE"


def find_anodic_sweeps(
    potential_v: pd.Series,
    tolerance_v: float,
) -> list[tuple[int, int]]:
    """
    Identify contiguous anodic (increasing-potential) sweeps.

    Returns a list of inclusive index ranges:
        [(start_index, end_index), ...]

    A preceding point is included in each sweep so that interpolation across
    the first increasing step remains possible.
    """
    potential = pd.to_numeric(potential_v, errors="coerce").to_numpy(dtype=float)

    if np.isnan(potential).any():
        raise ValueError(
            f"The '{POTENTIAL_COLUMN}' column contains missing or nonnumeric values."
        )

    differences = np.diff(potential)
    increasing = differences > tolerance_v

    sweeps: list[tuple[int, int]] = []
    position = 0

    while position < len(increasing):
        if not increasing[position]:
            position += 1
            continue

        run_start = position
        while position + 1 < len(increasing) and increasing[position + 1]:
            position += 1
        run_end = position

        # Difference index k represents the step from row k to row k + 1.
        start_index = run_start
        end_index = run_end + 1
        sweeps.append((start_index, end_index))

        position += 1

    if not sweeps:
        raise ValueError(
            "No anodic sweeps were detected from the potential column."
        )

    return sweeps


def first_confirmed_threshold_index(
    current_density: np.ndarray,
    target: float,
    minimum_consecutive_points: int,
) -> int | None:
    """
    Return the first index beginning a confirmed run at or above the target.
    """
    if minimum_consecutive_points < 1:
        raise ValueError("MIN_CONSECUTIVE_POINTS must be at least 1.")

    at_or_above = np.isfinite(current_density) & (current_density >= target)

    if minimum_consecutive_points == 1:
        matching = np.flatnonzero(at_or_above)
        return int(matching[0]) if len(matching) else None

    run_length = 0
    for index, is_above in enumerate(at_or_above):
        if is_above:
            run_length += 1
            if run_length >= minimum_consecutive_points:
                return index - minimum_consecutive_points + 1
        else:
            run_length = 0

    return None


def interpolate_potential_at_target(
    potential_v: np.ndarray,
    current_density: np.ndarray,
    target_current_density: float,
    minimum_consecutive_points: int,
) -> float:
    """
    Calculate the potential at the target current density.

    The first point that begins the required consecutive run at or above the
    target is identified. Linear interpolation is then performed between that
    point and the nearest preceding finite point below the target.

    Returns NaN when the target is not reached during the sweep.
    """
    valid = np.isfinite(potential_v) & np.isfinite(current_density)
    potential = potential_v[valid]
    current = current_density[valid]

    if len(potential) < 2:
        return float("nan")

    crossing_index = first_confirmed_threshold_index(
        current_density=current,
        target=target_current_density,
        minimum_consecutive_points=minimum_consecutive_points,
    )

    if crossing_index is None:
        return float("nan")

    if crossing_index == 0:
        return float(potential[0])

    previous_candidates = np.flatnonzero(
        current[:crossing_index] < target_current_density
    )
    if len(previous_candidates) == 0:
        return float(potential[crossing_index])

    lower_index = int(previous_candidates[-1])
    upper_index = int(crossing_index)

    lower_current = float(current[lower_index])
    upper_current = float(current[upper_index])
    lower_potential = float(potential[lower_index])
    upper_potential = float(potential[upper_index])

    if np.isclose(upper_current, lower_current):
        return upper_potential

    fraction = (
        (target_current_density - lower_current)
        / (upper_current - lower_current)
    )

    return lower_potential + fraction * (
        upper_potential - lower_potential
    )


def calculate_channel_overpotentials(
    dataframe: pd.DataFrame,
    channel: str,
    anodic_sweeps: list[tuple[int, int]],
    target_current_density: float,
    theoretical_potential_v: float,
    minimum_consecutive_points: int,
) -> list[float]:
    """Calculate one overpotential value for each anodic sweep."""
    current_density = pd.to_numeric(
        dataframe[channel],
        errors="coerce",
    ).to_numpy(dtype=float)

    potential = pd.to_numeric(
        dataframe[POTENTIAL_COLUMN],
        errors="coerce",
    ).to_numpy(dtype=float)

    overpotentials_mv: list[float] = []

    for start_index, end_index in anodic_sweeps:
        sweep_potential = potential[start_index : end_index + 1]
        sweep_current = current_density[start_index : end_index + 1]

        target_potential_v = interpolate_potential_at_target(
            potential_v=sweep_potential,
            current_density=sweep_current,
            target_current_density=target_current_density,
            minimum_consecutive_points=minimum_consecutive_points,
        )

        if np.isnan(target_potential_v):
            overpotentials_mv.append(float("nan"))
        else:
            overpotential_mv = (
                target_potential_v - theoretical_potential_v
            ) * 1000.0
            overpotentials_mv.append(float(overpotential_mv))

    return overpotentials_mv


def calculate_overpotential_table(
    input_csv: Path,
    output_csv: Path,
    target_current_density: float,
    theoretical_potential_v: float,
    minimum_consecutive_points: int,
) -> Path:
    """
    Read a combined current-density CSV and export cycle-resolved overpotentials.
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV was not found: {input_csv}")

    dataframe = pd.read_csv(input_csv, encoding="utf-8-sig")

    required_columns = {TIME_COLUMN, POTENTIAL_COLUMN}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"Input CSV is missing required columns: {sorted(missing_columns)}"
        )

    channel_columns = [
        column
        for column in dataframe.columns
        if column.lower().startswith("ch")
    ]
    if not channel_columns:
        raise ValueError(
            "No channel columns named ch1, ch2, ... were found."
        )

    channel_columns = sorted(
        channel_columns,
        key=lambda name: int(name[2:]) if name[2:].isdigit() else float("inf"),
    )

    anodic_sweeps = find_anodic_sweeps(
        potential_v=dataframe[POTENTIAL_COLUMN],
        tolerance_v=POTENTIAL_STEP_TOLERANCE_V,
    )

    results: list[dict[str, float | str | int]] = []

    for channel in channel_columns:
        cycle_values = calculate_channel_overpotentials(
            dataframe=dataframe,
            channel=channel,
            anodic_sweeps=anodic_sweeps,
            target_current_density=target_current_density,
            theoretical_potential_v=theoretical_potential_v,
            minimum_consecutive_points=minimum_consecutive_points,
        )

        finite_values = np.asarray(
            [value for value in cycle_values if np.isfinite(value)],
            dtype=float,
        )

        result: dict[str, float | str | int] = {
            "channel": channel,
        }

        for cycle_number, value in enumerate(cycle_values, start=1):
            result[f"cycle_{cycle_number}_overpotential_mV"] = value

        result["valid_cycle_count"] = int(len(finite_values))
        result["mean_overpotential_mV"] = (
            float(np.mean(finite_values))
            if len(finite_values) > 0
            else float("nan")
        )
        result["std_overpotential_mV"] = (
            float(np.std(finite_values, ddof=1))
            if len(finite_values) > 1
            else float("nan")
        )

        results.append(result)

    result_dataframe = pd.DataFrame(results)

    # Replace unresolved threshold values with the explicit text "n.d."
    # only in the exported CSV. Internal calculations remain numeric.
    export_dataframe = result_dataframe.astype(object).where(
        pd.notna(result_dataframe),
        "n.d.",
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    export_dataframe.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Input CSV: {input_csv}")
    print(f"Detected anodic sweeps: {len(anodic_sweeps)}")
    print(
        f"Target current density: "
        f"{target_current_density} mA/cm^2"
    )
    print(
        f"Theoretical potential: "
        f"{theoretical_potential_v} V vs RHE"
    )
    print(
        f"Consecutive-point requirement: "
        f"{minimum_consecutive_points}"
    )
    print("Overpotential unit: mV")
    print(f"Output CSV: {output_csv}")

    return output_csv


def main() -> None:
    """Run the overpotential calculation using the settings above."""
    calculate_overpotential_table(
        input_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV,
        target_current_density=TARGET_CURRENT_DENSITY_MA_CM2,
        theoretical_potential_v=THEORETICAL_POTENTIAL_V_VS_RHE,
        minimum_consecutive_points=MIN_CONSECUTIVE_POINTS,
    )


if __name__ == "__main__":
    main()
