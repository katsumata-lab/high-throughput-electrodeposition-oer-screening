from __future__ import annotations

import csv
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from potentiostat import Potentiostat


# =======================================================
# Input configuration
# =======================================================
CONDITIONS_CSV = Path("input/conditions/electrodeposition_conditions_96cond_1.csv")

HUB_COLUMN = "hub"
PORT_COLUMN = "port"


# =======================================================
# Output configuration
# =======================================================
OUTPUT_BASE_DIR = Path("output/OER")


# =======================================================
# Data acquisition settings
# =======================================================
SAMPLE_RATE_HZ = 10
CV_CURRENT_RANGE = "1000uA"


# =======================================================
# Pre-measurement stabilization
# =======================================================
# This is a waiting period before CV measurement, not an OCP measurement.
PRE_OER_STABILIZATION_MIN = 10


# =======================================================
# Cyclic voltammetry settings for OER evaluation
# =======================================================
CV_START_POTENTIAL_V = 0.0
CV_END_POTENTIAL_V = 1.0
CV_SCAN_RATE_MV_S = 10
CV_NUMBER_OF_CYCLES = 3

CV_QUIET_POTENTIAL_V = 0.0
CV_QUIET_TIME_MS = 1000


# =======================================================
# Parallel execution settings
# =======================================================
N_WORKER_PROCESSES = 2
MAX_THREADS_PER_PROCESS = 16

PRINT_EACH_RESULT = True
CONNECTION_SETTLING_TIME_S = 0.15

# Small delays reduce transient USB communication load.
DEVICE_START_STAGGER_S = 0.05
WORKER_START_JITTER_S = 0.20

CONNECTION_RETRIES = 2
MEASUREMENT_RETRIES = 1
RETRY_BACKOFF_S = 0.5
POST_ERROR_DELAY_S = 0.10


def load_ports_grouped_by_hub(csv_path: Path) -> dict[int, list[str]]:
    """Load unique Rodeostat COM ports and group them by USB hub."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Configuration CSV was not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("The CSV header could not be read.")

        required_columns = {HUB_COLUMN, PORT_COLUMN}
        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"The CSV is missing required columns {sorted(missing_columns)}. "
                f"Available columns: {reader.fieldnames}"
            )

        ports_by_hub: dict[int, list[str]] = {}
        observed_ports: set[str] = set()

        for line_number, row in enumerate(reader, start=2):
            port = (row.get(PORT_COLUMN, "") or "").strip()
            if not port:
                continue
            if port in observed_ports:
                raise ValueError(
                    f"Duplicate COM port at line {line_number}: {port}"
                )
            observed_ports.add(port)

            try:
                hub_id = int(float(row.get(HUB_COLUMN, "")))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Hub ID must be an integer at line {line_number}: "
                    f"{row.get(HUB_COLUMN)!r}"
                ) from error

            if hub_id <= 0:
                raise ValueError(
                    f"Hub ID must be a positive integer at line {line_number}: "
                    f"{hub_id}"
                )

            ports_by_hub.setdefault(hub_id, []).append(port)

    if not ports_by_hub:
        raise ValueError(
            f"No valid values were found in the '{HUB_COLUMN}' and "
            f"'{PORT_COLUMN}' columns."
        )

    for hub_id in ports_by_hub:
        ports_by_hub[hub_id] = sorted(ports_by_hub[hub_id])

    return ports_by_hub


def make_safe_port_tag(port: str) -> str:
    """Convert a COM port identifier into a filename-safe string."""
    return port.replace("/", "_").replace("\\", "_").replace(":", "_")


def calculate_cv_offset_and_amplitude(
    start_potential_v: float,
    end_potential_v: float,
) -> tuple[float, float]:
    """Convert CV start/end potentials to Rodeostat offset and amplitude."""
    lower_potential_v = min(start_potential_v, end_potential_v)
    upper_potential_v = max(start_potential_v, end_potential_v)
    offset_v = 0.5 * (upper_potential_v + lower_potential_v)
    amplitude_v = 0.5 * (upper_potential_v - lower_potential_v)
    return offset_v, amplitude_v


def calculate_cv_period_ms(
    amplitude_v: float,
    scan_rate_v_s: float,
) -> int:
    """Calculate the triangular-wave period required by the Rodeostat API."""
    if scan_rate_v_s <= 0:
        raise ValueError("The CV scan rate must be greater than zero.")
    period_s = 4.0 * amplitude_v / scan_rate_v_s
    return int(period_s * 1000)


def wait_until_all_devices_are_ready(
    ready_count: mp.Value,
    ready_condition: mp.Condition,
    total_devices: int,
    polling_interval_s: float = 0.5,
) -> None:
    """Block until every assigned Rodeostat has connected and initialized."""
    with ready_condition:
        while ready_count.value < total_devices:
            ready_condition.wait(timeout=polling_interval_s)


def connect_with_retry(port: str) -> Potentiostat:
    """Connect to one Rodeostat, retrying failed attempts."""
    last_error: Exception | None = None

    for attempt in range(CONNECTION_RETRIES + 1):
        try:
            return Potentiostat(port)
        except Exception as error:
            last_error = error
            time.sleep(RETRY_BACKOFF_S * (2**attempt))

    raise RuntimeError(
        f"Failed to connect to the potentiostat on {port} after retries: "
        f"{last_error}"
    )


def run_measurement_with_retry(
    device: Potentiostat,
    test_name: str,
    parameters: dict,
    output_filename: str,
) -> None:
    """Run a Rodeostat measurement and retry if execution fails."""
    last_error: Exception | None = None

    for attempt in range(MEASUREMENT_RETRIES + 1):
        try:
            try:
                device.set_param(test_name, parameters)
            except Exception:
                # Some Rodeostat library versions accept parameters only in run_test().
                pass

            try:
                device.run_test(
                    test_name,
                    filename=output_filename,
                    timeunit="s",
                )
                return
            except TypeError:
                # Compatibility fallback for library versions requiring param=.
                device.run_test(
                    test_name,
                    param=parameters,
                    filename=output_filename,
                    timeunit="s",
                )
                return

        except Exception as error:
            last_error = error
            time.sleep(RETRY_BACKOFF_S * (2**attempt))

    raise RuntimeError(
        f"Measurement '{test_name}' failed after retries: {last_error}"
    )


def run_oer_cv_on_device(
    port: str,
    channel_label: str,
    output_directory: Path,
    measurement_start_event: mp.Event,
    ready_count: mp.Value,
    ready_condition: mp.Condition,
    start_delay_s: float,
) -> tuple[str, bool, str]:
    """Connect to one Rodeostat and run one OER CV measurement."""
    device: Potentiostat | None = None

    try:
        device = connect_with_retry(port)

        try:
            device.set_sample_rate(SAMPLE_RATE_HZ)
        except Exception:
            pass

        try:
            device.set_volt(0.0)
        except Exception:
            pass

        time.sleep(CONNECTION_SETTLING_TIME_S)

        with ready_condition:
            ready_count.value += 1
            ready_condition.notify_all()

        # Wait until the main process completes the stabilization period.
        measurement_start_event.wait()

        # Stagger starts slightly to reduce transient USB communication load.
        if start_delay_s > 0:
            time.sleep(start_delay_s)

        try:
            device.set_curr_range(CV_CURRENT_RANGE)
        except Exception:
            pass

        test_name = "cyclic"
        default_parameters = device.get_param(test_name)

        offset_v, amplitude_v = calculate_cv_offset_and_amplitude(
            CV_START_POTENTIAL_V,
            CV_END_POTENTIAL_V,
        )
        scan_rate_v_s = CV_SCAN_RATE_MV_S / 1000.0
        period_ms = calculate_cv_period_ms(amplitude_v, scan_rate_v_s)

        cv_parameters = default_parameters.copy()
        cv_parameters["quietValue"] = float(CV_QUIET_POTENTIAL_V)
        cv_parameters["quietTime"] = int(CV_QUIET_TIME_MS)
        cv_parameters["amplitude"] = float(amplitude_v)
        cv_parameters["offset"] = float(offset_v)
        cv_parameters["period"] = int(period_ms)
        cv_parameters["numCycles"] = int(CV_NUMBER_OF_CYCLES)
        cv_parameters["shift"] = 0.0

        output_path = output_directory / (
            f"{channel_label}_cv_"
            f"{CV_START_POTENTIAL_V:+.3f}to{CV_END_POTENTIAL_V:+.3f}V_"
            f"{CV_SCAN_RATE_MV_S}mVs_{CV_NUMBER_OF_CYCLES}cyc.csv"
        )

        run_measurement_with_retry(
            device,
            test_name,
            cv_parameters,
            str(output_path),
        )

        try:
            device.set_param(test_name, default_parameters)
        except Exception:
            pass

        try:
            device.set_volt(0.0)
        except Exception:
            pass

        return (
            channel_label,
            True,
            f"CV completed on {port}: {output_path.name}",
        )

    except Exception as error:
        time.sleep(POST_ERROR_DELAY_S)
        return (
            channel_label,
            False,
            f"Error on {port}: {error}",
        )

    finally:
        if device is not None:
            try:
                device.close()
            except Exception:
                pass


def run_worker_process(
    worker_id: int,
    assigned_ports: list[str],
    output_directory: Path,
    measurement_start_event: mp.Event,
    ready_count: mp.Value,
    ready_condition: mp.Condition,
) -> None:
    """Run device-specific OER CV tasks concurrently within one process."""
    if not assigned_ports:
        return

    thread_count = min(MAX_THREADS_PER_PROCESS, len(assigned_ports))
    if PRINT_EACH_RESULT:
        print(
            f"[worker{worker_id:02d}] "
            f"ports={len(assigned_ports)} threads={thread_count}"
        )

    futures = []
    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        for device_index, port in enumerate(assigned_ports, start=1):
            channel_label = (
                f"w{worker_id:02d}_d{device_index:02d}_{make_safe_port_tag(port)}"
            )
            start_delay_s = (
                (worker_id - 1) * WORKER_START_JITTER_S
                + (device_index - 1) * DEVICE_START_STAGGER_S
            )

            futures.append(
                executor.submit(
                    run_oer_cv_on_device,
                    port,
                    channel_label,
                    output_directory,
                    measurement_start_event,
                    ready_count,
                    ready_condition,
                    start_delay_s,
                )
            )

        for future in as_completed(futures):
            channel_label, succeeded, message = future.result()
            if PRINT_EACH_RESULT:
                status = "OK" if succeeded else "FAILED"
                print(f"[{channel_label}] {status} | {message}")


def main() -> None:
    """Load the channel configuration and run parallel OER CV measurements."""
    ports_by_hub = load_ports_grouped_by_hub(CONDITIONS_CSV)

    sorted_hub_ids = sorted(ports_by_hub)
    hub_count = len(sorted_hub_ids)
    device_count = sum(len(ports_by_hub[hub_id]) for hub_id in sorted_hub_ids)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = OUTPUT_BASE_DIR / f"run_{run_timestamp}"
    output_directory.mkdir(parents=True, exist_ok=True)

    context = mp.get_context("spawn")
    measurement_start_event = context.Event()
    ready_count = context.Value("i", 0)
    ready_lock = context.Lock()
    ready_condition = context.Condition(ready_lock)

    process_count = min(N_WORKER_PROCESSES, hub_count)
    ports_by_worker: list[list[str]] = [[] for _ in range(process_count)]
    hubs_by_worker: list[list[int]] = [[] for _ in range(process_count)]

    for hub_index, hub_id in enumerate(sorted_hub_ids):
        worker_slot = hub_index % process_count
        ports_by_worker[worker_slot].extend(ports_by_hub[hub_id])
        hubs_by_worker[worker_slot].append(hub_id)

    if PRINT_EACH_RESULT:
        for worker_slot in range(process_count):
            print(
                f"[MAIN] worker_slot={worker_slot} "
                f"hubs={hubs_by_worker[worker_slot]} "
                f"ports={len(ports_by_worker[worker_slot])}"
            )

    processes: list[mp.Process] = []
    for worker_slot in range(process_count):
        assigned_ports = ports_by_worker[worker_slot]
        if not assigned_ports:
            continue

        worker_id = hubs_by_worker[worker_slot][0]
        process = context.Process(
            target=run_worker_process,
            args=(
                worker_id,
                assigned_ports,
                output_directory,
                measurement_start_event,
                ready_count,
                ready_condition,
            ),
            daemon=False,
        )
        process.start()
        processes.append(process)

    print(
        f"[MAIN] Waiting until all devices are ready... "
        f"({device_count} devices)"
    )
    wait_until_all_devices_are_ready(
        ready_count,
        ready_condition,
        total_devices=device_count,
        polling_interval_s=0.5,
    )
    print(
        f"[MAIN] All devices are ready. "
        f"(ready={ready_count.value}/{device_count})"
    )

    stabilization_time_s = int(PRE_OER_STABILIZATION_MIN * 60)
    print(
        f"[MAIN] Waiting {PRE_OER_STABILIZATION_MIN} min "
        f"({stabilization_time_s} s) before starting CV measurements..."
    )
    time.sleep(stabilization_time_s)

    print("[MAIN] Starting concurrent CV measurements.")
    measurement_start_event.set()

    for process in processes:
        process.join()

    print(f"All measurements completed.\nOutput folder: {output_directory}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
