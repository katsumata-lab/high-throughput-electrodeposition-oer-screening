from __future__ import annotations

import csv
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from potentiostat import Potentiostat

# =======================================================
# User settings
# =======================================================
PRE_DEPOSITION_STABILIZATION_MIN = 20
CONDITIONS_CSV = Path("input/conditions/electrodeposition_conditions_96cond_1.csv")
OUTPUT_BASE_DIR = Path("output/ECD")

# =======================================================
# Parallel-execution settings for the 32-channel setup
# =======================================================
N_WORKER_PROCESSES = 2
MAX_THREADS_PER_PROCESS = 16

PRINT_EACH_RESULT = True
CONNECTION_SETTLING_TIME_S = 0.15

# Small offsets reduce transient USB communication load.
DEVICE_START_STAGGER_S = 0.05
WORKER_START_OFFSET_S = 0.20

CONNECTION_RETRIES = 2
MEASUREMENT_RETRIES = 1
RETRY_BACKOFF_S = 0.5

POST_ERROR_DELAY_S = 0.10
SAMPLE_RATE_HZ = 10
CURRENT_RANGE = "1000uA"

# Required column names in the condition CSV file.
HUB_COLUMN = "hub"
PORT_COLUMN = "port"
POTENTIAL_COLUMN = "CA_VOLT"
TIME_COLUMN = "CA_TIME_S"


DepositionCondition = dict[str, float | int]
DepositionPlan = dict[str, DepositionCondition]
HubPortMap = dict[int, list[str]]


def load_deposition_plan(csv_path: Path) -> tuple[DepositionPlan, HubPortMap]:
    """Load channel-specific electrodeposition conditions from a CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Condition CSV file was not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("The CSV header could not be read.")

        required_columns = {
            HUB_COLUMN,
            PORT_COLUMN,
            POTENTIAL_COLUMN,
            TIME_COLUMN,
        }
        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"The CSV file is missing required columns {sorted(missing_columns)}. "
                f"Available columns: {reader.fieldnames}"
            )

        deposition_plan: DepositionPlan = {}
        hub_port_map: HubPortMap = {}

        for line_number, row in enumerate(reader, start=2):
            port = (row.get(PORT_COLUMN, "") or "").strip()
            if not port:
                raise ValueError(f"Line {line_number}: '{PORT_COLUMN}' is empty.")
            if port in deposition_plan:
                raise ValueError(f"Line {line_number}: duplicate port entry: {port}")

            try:
                hub_id = int(float(row[HUB_COLUMN]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Line {line_number}: '{HUB_COLUMN}' must be an integer: "
                    f"{row.get(HUB_COLUMN)!r}"
                ) from exc
            if hub_id <= 0:
                raise ValueError(
                    f"Line {line_number}: '{HUB_COLUMN}' must be a positive integer: {hub_id}"
                )

            try:
                deposition_potential_v = float(row[POTENTIAL_COLUMN])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Line {line_number}: '{POTENTIAL_COLUMN}' must be numeric: "
                    f"{row.get(POTENTIAL_COLUMN)!r}"
                ) from exc

            try:
                deposition_time_s = int(float(row[TIME_COLUMN]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Line {line_number}: '{TIME_COLUMN}' must be an integer number of seconds: "
                    f"{row.get(TIME_COLUMN)!r}"
                ) from exc
            if deposition_time_s <= 0:
                raise ValueError(
                    f"Line {line_number}: '{TIME_COLUMN}' must be positive: {deposition_time_s}"
                )

            deposition_plan[port] = {
                "hub_id": hub_id,
                "deposition_potential_v": deposition_potential_v,
                "deposition_time_s": deposition_time_s,
            }
            hub_port_map.setdefault(hub_id, []).append(port)

    for hub_id in hub_port_map:
        hub_port_map[hub_id] = sorted(hub_port_map[hub_id])

    return deposition_plan, hub_port_map


def make_port_filename_tag(port: str) -> str:
    """Convert a serial-port name into a filesystem-safe tag."""
    return port.replace("/", "_").replace("\\", "_").replace(":", "_")


def wait_until_all_devices_ready(
    ready_count: mp.Value,
    ready_condition: mp.Condition,
    total_devices: int,
    polling_interval_s: float = 0.5,
) -> None:
    """Block until all requested Rodeostat units report successful connection."""
    with ready_condition:
        while ready_count.value < total_devices:
            ready_condition.wait(timeout=polling_interval_s)


def connect_with_retry(port: str) -> Potentiostat:
    """Connect to one Rodeostat, retrying transient failures."""
    last_error: Exception | None = None
    for attempt in range(CONNECTION_RETRIES + 1):
        try:
            return Potentiostat(port)
        except Exception as exc:
            last_error = exc
            time.sleep(RETRY_BACKOFF_S * (2**attempt))

    raise RuntimeError(
        f"Failed to connect to the potentiostat on {port} after retries: {last_error}"
    )


def run_test_with_retry(
    device: Potentiostat,
    test_name: str,
    parameters: dict,
    output_filename: str,
) -> None:
    """Run one Rodeostat test, retrying transient execution failures."""
    last_error: Exception | None = None
    for attempt in range(MEASUREMENT_RETRIES + 1):
        try:
            device.run_test(
                test_name,
                param=parameters,
                filename=output_filename,
                timeunit="s",
            )
            return
        except Exception as exc:
            last_error = exc
            time.sleep(RETRY_BACKOFF_S * (2**attempt))

    raise RuntimeError(f"The '{test_name}' test failed after retries: {last_error}")


def run_single_channel(
    port: str,
    channel_label: str,
    output_directory: Path,
    deposition_potential_v: float,
    deposition_time_s: int,
    pre_deposition_wait_s: int,
    start_event: mp.Event,
    ready_count: mp.Value,
    ready_condition: mp.Condition,
    start_offset_s: float,
) -> tuple[str, bool, str]:
    """Connect to and run the electrodeposition sequence for one channel."""
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

        # Wait until every connected device is ready to enter the shared timeline.
        start_event.wait()

        # Stagger device starts slightly to reduce transient USB communication load.
        if start_offset_s > 0:
            time.sleep(start_offset_s)

        # Adjust the pre-deposition wait so channels with different deposition
        # durations finish at approximately the same time.
        if pre_deposition_wait_s > 0:
            time.sleep(pre_deposition_wait_s)

        try:
            device.set_curr_range(CURRENT_RANGE)
        except Exception:
            pass

        test_name = "chronoamp"
        default_parameters = device.get_param(test_name)

        deposition_time_ms = int(deposition_time_s * 1000)
        deposition_parameters = default_parameters.copy()
        deposition_parameters["quietValue"] = 0.0
        deposition_parameters["quietTime"] = 0
        deposition_parameters["step"] = [
            (deposition_time_ms, float(deposition_potential_v))
        ]

        output_path = output_directory / (
            f"{channel_label}_ca_{deposition_potential_v:+.2f}V_"
            f"{deposition_time_s}s.csv"
        )
        run_test_with_retry(
            device,
            test_name,
            deposition_parameters,
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
            f"Electrodeposition completed on {port} "
            f"(wait={pre_deposition_wait_s}s, deposition={deposition_time_s}s) "
            f"-> {output_path.name}",
        )

    except Exception as exc:
        time.sleep(POST_ERROR_DELAY_S)
        return channel_label, False, f"ERROR on {port}: {exc}"

    finally:
        if device is not None:
            try:
                device.close()
            except Exception:
                pass


def run_worker_process(
    worker_id: int,
    assigned_ports: list[str],
    deposition_plan: DepositionPlan,
    output_directory: Path,
    start_event: mp.Event,
    ready_count: mp.Value,
    ready_condition: mp.Condition,
    base_stabilization_time_s: int,
    maximum_deposition_time_s: int,
) -> None:
    """Run the channels assigned to one worker process using device-level threads."""
    if not assigned_ports:
        return

    max_workers = min(MAX_THREADS_PER_PROCESS, len(assigned_ports))
    if PRINT_EACH_RESULT:
        print(
            f"[worker{worker_id:02d}] ports={len(assigned_ports)} "
            f"threads={max_workers}"
        )

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for device_index, port in enumerate(assigned_ports, start=1):
            condition = deposition_plan[port]
            deposition_potential_v = float(condition["deposition_potential_v"])
            deposition_time_s = int(condition["deposition_time_s"])
            channel_label = (
                f"w{worker_id:02d}_d{device_index:02d}_{make_port_filename_tag(port)}"
            )

            start_offset_s = (
                (worker_id - 1) * WORKER_START_OFFSET_S
                + (device_index - 1) * DEVICE_START_STAGGER_S
            )

            # Keep the total timeline approximately constant across channels:
            # total time = base stabilization time + maximum deposition time.
            pre_deposition_wait_s = base_stabilization_time_s + max(
                0,
                maximum_deposition_time_s - deposition_time_s,
            )

            futures.append(
                executor.submit(
                    run_single_channel,
                    port,
                    channel_label,
                    output_directory,
                    deposition_potential_v,
                    deposition_time_s,
                    pre_deposition_wait_s,
                    start_event,
                    ready_count,
                    ready_condition,
                    start_offset_s,
                )
            )

        for future in as_completed(futures):
            channel_label, succeeded, message = future.result()
            if PRINT_EACH_RESULT:
                status = "OK" if succeeded else "FAILED"
                print(f"[{channel_label}] {status} | {message}")


def main() -> None:
    deposition_plan, hub_port_map = load_deposition_plan(CONDITIONS_CSV)

    ports = list(deposition_plan)
    number_of_ports = len(ports)
    if number_of_ports == 0:
        print("[ERROR] No serial ports were specified in the condition CSV file.")
        return

    maximum_deposition_time_s = max(
        int(deposition_plan[port]["deposition_time_s"]) for port in ports
    )
    base_stabilization_time_s = int(PRE_DEPOSITION_STABILIZATION_MIN * 60)
    target_total_time_s = base_stabilization_time_s + maximum_deposition_time_s

    sorted_hub_ids = sorted(hub_port_map)
    number_of_hubs = len(sorted_hub_ids)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = OUTPUT_BASE_DIR / f"run_{run_tag}"
    output_directory.mkdir(parents=True, exist_ok=True)

    multiprocessing_context = mp.get_context("spawn")
    start_event = multiprocessing_context.Event()

    ready_count = multiprocessing_context.Value("i", 0)
    ready_lock = multiprocessing_context.Lock()
    ready_condition = multiprocessing_context.Condition(ready_lock)

    number_of_processes = min(N_WORKER_PROCESSES, number_of_hubs)
    port_assignments: list[list[str]] = [
        [] for _ in range(number_of_processes)
    ]
    hub_assignments: list[list[int]] = [
        [] for _ in range(number_of_processes)
    ]

    for hub_index, hub_id in enumerate(sorted_hub_ids):
        worker_index = hub_index % number_of_processes
        port_assignments[worker_index].extend(hub_port_map[hub_id])
        hub_assignments[worker_index].append(hub_id)

    if PRINT_EACH_RESULT:
        print(
            f"[MAIN] base_stabilization={base_stabilization_time_s}s, "
            f"max_deposition={maximum_deposition_time_s}s, "
            f"target_total={target_total_time_s}s"
        )
        for worker_index in range(number_of_processes):
            assigned_hubs = hub_assignments[worker_index]
            assigned_port_count = len(port_assignments[worker_index])
            print(
                f"[MAIN] worker_slot={worker_index} hubs={assigned_hubs} "
                f"ports={assigned_port_count}"
            )

    processes: list[mp.Process] = []
    for worker_index in range(number_of_processes):
        assigned_ports = port_assignments[worker_index]
        if not assigned_ports:
            continue

        worker_id = hub_assignments[worker_index][0]
        process = multiprocessing_context.Process(
            target=run_worker_process,
            args=(
                worker_id,
                assigned_ports,
                deposition_plan,
                output_directory,
                start_event,
                ready_count,
                ready_condition,
                base_stabilization_time_s,
                maximum_deposition_time_s,
            ),
            daemon=False,
        )
        process.start()
        processes.append(process)

    print(f"[MAIN] Waiting until all devices are ready... ({number_of_ports} devices)")
    wait_until_all_devices_ready(
        ready_count,
        ready_condition,
        total_devices=number_of_ports,
        polling_interval_s=0.5,
    )
    print(
        f"[MAIN] All devices are ready. "
        f"(ready={ready_count.value}/{number_of_ports})"
    )

    # Start the shared experimental timeline. Each channel then waits for its
    # individually adjusted stabilization period before electrodeposition.
    print("[MAIN] Starting the channel-specific waiting timeline.")
    start_event.set()

    for process in processes:
        process.join()

    print(f"All operations completed. Output folder: {output_directory}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
