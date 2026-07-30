# High-Throughput Electrodeposition and OER Screening Platform

This repository contains the Python programs, experimental-condition files,
processed screening data, and visualization scripts for the high-throughput
hydroxide electrodeposition and oxygen evolution reaction (OER) evaluation
platform described in the associated manuscript:

> **High-Throughput Experimentation Platform for Hydroxide Electrodeposition
> and Electrochemical Evaluation of Electrocatalysts**

The platform controls multiple open-source Rodeostat potentiostats concurrently
and performs channel-specific electrodeposition followed by parallel
cyclic-voltammetry (CV)-based OER evaluation.

## Repository contents

```text
.
├── identify_com_ports.py
├── parallel_electrodeposition.py
├── parallel_oer_measurement.py
├── combine_OER_data.py
├── calculate_overpotential.py
├── plot_overpotential_heatmap.py
├── visualize_hte_decision_tree.py
├── decision_tree_pie_viz.py
├── LOOCV_decision_tree.py
├── requirements.txt
├── input/
│   ├── conditions/
│   │   ├── electrodeposition_conditions_96cond_1.csv
│   │   ├── electrodeposition_conditions_96cond_2.csv
│   │   └── electrodeposition_conditions_96cond_3.csv
│   └── results/
│       └── 96_conditions_results.csv
└── output/                 # Created automatically when scripts are run
```

Each condition file contains one 32-channel batch: 16 channels on each of two
USB hubs. Together, the three files describe the complete 96-condition screen.
The control scripts use `electrodeposition_conditions_96cond_1.csv` by default.
Edit `CONDITIONS_CSV`, or `CSV_PATH` in `identify_com_ports.py`, before running
batches 2 or 3.

## Experimental and software environment

- Operating system: Windows
- Python: 3.12
- Potentiostat: Rodeostat HC (high-current configuration)
- Firmware: unmodified IO Rodeo firmware `FW0.1.0`
- Firmware source revision:
  `86e4708fea84f8fc33bcbfc9a706b06f4b770efd`
- High-current firmware variant: `CURRENT_VARIANT_MILL10_AMP`
  (`10MilliAmp`)
- Available current ranges for this variant:
  ±10, ±100, ±1000, and ±10000 µA
- Parallel configuration used for each batch: 32 Rodeostat units
- USB connection: two externally powered USB hubs, 16 units per hub
- Concurrency: two worker processes, with up to 16 device-specific threads per
  process
- Sampling rate: 10 Hz

No Rodeostat circuit or firmware modifications were made. The devices are
operated concurrently rather than being triggered at an exactly identical
instant. Small device-dependent delays are introduced to reduce transient USB
communication load.

As of 30 July 2026, `FW0.1.0` is the latest firmware version present in the
official IO Rodeo repository. Users reproducing the experiments should verify
that the firmware and hardware variant loaded on their own devices are
compatible with the settings used here.

## Installation

Install the required Python packages from the repository root:

```bash
pip install -r requirements.txt
```

The supplied `requirements.txt` contains:

```text
numpy
pandas
matplotlib
scikit-learn
iorodeo-potentiostat
```

The package `iorodeo-potentiostat` provides:

```python
from potentiostat import Potentiostat
```

Official resources:

- Rodeostat repository:
  <https://github.com/iorodeo/potentiostat>
- Python-library documentation:
  <https://iorodeo.github.io/iorodeo-potentiostat-docs-build/>
- Firmware programming guide:
  <https://blog.iorodeo.com/rodeostat-firmware-programming-instructions/>

Run all commands below from the repository root.

## Channel identification

Before an experiment, confirm the correspondence between each physical
Rodeostat and its Windows COM port:

```bash
python identify_com_ports.py
```

The script reads
`input/conditions/electrodeposition_conditions_96cond_1.csv` by default and
sequentially performs a 1 s constant-potential test on the listed ports. The
active Rodeostat is identified visually from the change in its onboard LED
state. Stop the script manually after completing the required mapping. Change
`CSV_PATH` before checking another batch.

## Input condition files

The files in `input/conditions/` contain one row per independently controlled
Rodeostat channel.

Required control columns are:

| Column | Description |
| --- | --- |
| `hub` | USB-hub group assigned to the Rodeostat |
| `port` | Windows COM port assigned to the Rodeostat |
| `CA_VOLT` | Electrodeposition potential (V) |
| `CA_TIME_S` | Electrodeposition duration (s) |

Additional columns, including `Ni/Fe` and `Support`, are experimental metadata
used to associate the electrochemical program with the electrolyte composition.
They are not used directly by the control program to set the Rodeostat.

## Parallel electrodeposition

Run:

```bash
python parallel_electrodeposition.py
```

The script:

1. reads the channel-specific condition CSV;
2. groups devices by USB hub and COM port;
3. connects to and initializes all assigned Rodeostats;
4. applies a 20 min pre-deposition stabilization period;
5. performs channel-specific chronoamperometric electrodeposition;
6. saves one current-time CSV for each channel.

The current range is set to 1000 µA, and data are acquired at 10 Hz.

For unequal deposition durations, the per-channel waiting times are adjusted so
that the stabilization-plus-deposition sequences finish at approximately the
same time.

Output files are stored under:

```text
output/ECD/run_YYYYMMDD_HHMMSS/
```

## Parallel OER measurement

Run:

```bash
python parallel_oer_measurement.py
```

After all Rodeostats have connected and initialized, the script applies a
10 min pre-OER stabilization period and then starts concurrent CV measurements.

Default CV settings are:

- potential range: -0.2 to 0.8 V versus the printed pseudo-Ag/AgCl reference
  electrode;
- scan rate: 10 mV s<sup>-1</sup>;
- number of cycles: 3;
- current range: 1000 µA;
- sampling rate: 10 Hz.

One output CSV is saved for each channel under:

```text
output/OER/run_YYYYMMDD_HHMMSS/
```


## OER data combination and preprocessing

After the channel-specific OER measurements have been completed, run:

```bash
python combine_OER_data.py
```

By default, the script automatically selects the latest
`output/OER/run_*` folder and reads:

```text
input/conditions/electrodeposition_conditions_96cond_1.csv
```

A specific run folder and condition file can be selected explicitly:

```bash
python combine_OER_data.py \
  --input-dir output/OER/run_YYYYMMDD_HHMMSS \
  --conditions input/conditions/electrodeposition_conditions_96cond_1.csv
```

The geometric working-electrode area can also be changed from its default value:

```bash
python combine_OER_data.py --area-cm2 0.0264
```

The row order of this condition file defines the channel labels `ch1`, `ch2`,
..., `ch32`. Each raw Rodeostat file is associated with a channel through the
COM-port value in the `port` column and the COM-port token contained in the raw
filename.

The raw Rodeostat CSV files do not contain column names. The script interprets
the first three columns as:

| Raw column | Assigned name | Unit |
| --- | --- | --- |
| 1 | `time_s` | s |
| 2 | `potential_V` | V versus the printed pseudo-Ag/AgCl reference |
| 3 | `current_uA` | µA |

The default processing parameters are:

```python
WORKING_ELECTRODE_AREA_CM2 = 0.0264
POTENTIAL_CORRECTION_TO_RHE_V = 1.169
ZERO_POINT_DURATION_S = 1.0
ZERO_POTENTIAL_TOLERANCE_V = 1.0e-6
```

For each channel, the current is converted to geometric current density as:

```text
current density (mA cm^-2)
= current (µA) / 1000 / working-electrode area (cm^2)
```

A channel-specific zero-current correction is then applied. The mean current
density measured during the initial 1 s waiting period at an unconverted
potential of 0 V is calculated and subtracted from every current-density value
in that channel.

The potential is converted to the RHE scale as:

```text
E (V vs RHE) = E (V vs printed pseudo-Ag/AgCl) + 1.169 V
```

The combined output is saved in the selected run folder as:

```text
combined_current_density.csv
```

Its columns are:

```text
time_s
potential_V_vs_RHE
ch1
ch2
...
ch32
```

The `ch1`–`ch32` columns contain zero-point-corrected current density in
mA cm<sup>-2</sup>. If channel files have different numbers of rows, the
longest valid file defines the output length and shorter channels are padded
with blank cells after their final recorded value. A warning is printed if the
overlapping time or potential axes differ between channels.

## Cycle-resolved overpotential calculation

After generating `combined_current_density.csv`, set its path near the top of
`calculate_overpotential.py`:

```python
INPUT_CSV = Path(
    "output/OER/run_YYYYMMDD_HHMMSS/combined_current_density.csv"
)
```

Then run:

```bash
python calculate_overpotential.py
```

The default analysis settings are:

```python
TARGET_CURRENT_DENSITY_MA_CM2 = 1.0
THEORETICAL_POTENTIAL_V_VS_RHE = 1.23
MIN_CONSECUTIVE_POINTS = 3
```

The program identifies each anodic, increasing-potential sweep and calculates
the potential at which each channel reaches 1.0 mA cm<sup>-2</sup>. A threshold
crossing is accepted only when at least three consecutive data points are at or
above the target current density. The target potential is obtained by linear
interpolation between the nearest preceding point below the threshold and the
first point of the accepted threshold run.

The overpotential is calculated for each cycle as:

```text
overpotential (mV)
= [E at 1.0 mA cm^-2 (V vs RHE) - 1.23 V] × 1000
```

Results are saved in the same run folder as:

```text
overpotential_at_target_current.csv
```

The output contains:

```text
channel
cycle_1_overpotential_mV
cycle_2_overpotential_mV
cycle_3_overpotential_mV
valid_cycle_count
mean_overpotential_mV
std_overpotential_mV
```

The mean is calculated from all valid cycles. The standard deviation is the
sample standard deviation (`ddof=1`) and is calculated when at least two valid
cycles are available. If a cycle does not reach the specified current-density
criterion, the corresponding value is reported as `n.d.`. If no valid cycles
are available, the mean and standard deviation are also reported as `n.d.`.

## Screening dataset

The processed 96-condition dataset used for Figures 3 and 4 is provided as:

```text
input/results/96_conditions_results.csv
```

Descriptor columns:

- `Ni/Fe`
- `KNO3`
- `potential(V)`
- `time(s)`

Target column:

- `overpotential(mV)`

The corresponding `standard_deviation` is also included.

## Reproduction of Figure 3: overpotential heat maps

Each panel is generated separately by fixing `Ni/Fe` and `KNO3` and plotting
the OER overpotential against electrodeposition potential and deposition time.

Run:

```bash
python plot_overpotential_heatmap.py input/results/96_conditions_results.csv --fix "Ni/Fe=1" --fix "KNO3=0"
python plot_overpotential_heatmap.py input/results/96_conditions_results.csv --fix "Ni/Fe=1" --fix "KNO3=0.33"
python plot_overpotential_heatmap.py input/results/96_conditions_results.csv --fix "Ni/Fe=3" --fix "KNO3=0"
python plot_overpotential_heatmap.py input/results/96_conditions_results.csv --fix "Ni/Fe=3" --fix "KNO3=0.33"
python plot_overpotential_heatmap.py input/results/96_conditions_results.csv --fix "Ni/Fe=9" --fix "KNO3=0"
python plot_overpotential_heatmap.py input/results/96_conditions_results.csv --fix "Ni/Fe=9" --fix "KNO3=0.33"
```

Images are saved under `output/results/` by default.

The six panels were assembled into the final composite Figure 3 using Microsoft
PowerPoint. PowerPoint was used only for panel arrangement and labeling; no
numerical transformation was performed during figure assembly.

## Reproduction of Figure 4: decision-tree visualization

Run:

```bash
python visualize_hte_decision_tree.py
```

This script trains:

```python
DecisionTreeRegressor(random_state=42, max_depth=3)
```

using the four descriptor columns and saves:

```text
output/results/decision_tree_pies.png
```

In the custom visualization:

- node size represents the number of samples reaching the node;
- colored sectors show the distribution of quantile-binned overpotentials;
- branch labels show the splitting conditions;
- terminal nodes show the predicted mean overpotential.

The depth-3 model is used for Figure 4 because it provides a compact,
interpretable representation of the screening data.

## Leave-one-out cross-validation

Run with the default repository paths:

```bash
python LOOCV_decision_tree.py
```

Alternatively, specify the paths explicitly:

```bash
python LOOCV_decision_tree.py \
  --input input/results/96_conditions_results.csv \
  --output output/decision_tree_loocv_predictions_None.csv
```

For each leave-one-out fold, one condition is retained as the test sample, a
decision-tree regressor is trained on the remaining valid conditions, and the
retained condition is predicted. After all folds, the script reports:

- mean absolute error (MAE);
- root mean squared error (RMSE);
- coefficient of determination (R<sup>2</sup>).

It also saves the descriptor values, measured overpotential, predicted
overpotential, residual, and absolute error for every fold.

The maximum tree depth is controlled by the variable `d` near the top of
`LOOCV_decision_tree.py`:

```python
d = None
```

The supplied setting `d = None` evaluates an unrestricted decision tree. To
evaluate the same depth-limited model used for Figure 4, change the setting to:

```python
d = 3
```

For the model-complexity comparison reported in the revised manuscript, run the
script separately with:

```python
d = 2
d = 3
d = 4
d = 5
d = None
```

Use a different output filename for each run so that the fold-by-fold
predictions are not overwritten. The depth-3 tree is retained for the main
visualization because of its interpretability, whereas the depth comparison is
used to quantify the influence of model complexity on predictive performance.

## Notes on reproducibility

- COM port assignments depend on the Windows computer and may change after USB
  devices are reconnected.
- Confirm the correspondence between each COM port and physical Rodeostat
  before each experimental series.
- Change the condition CSV path at the top of the control script before running
  another 32-condition batch.
- Output directories are created automatically.
- Verify current ranges, potential limits, USB-hub capacity, electrode
  connections, and firmware compatibility before operating a modified hardware
  configuration.

## License

The original code in this repository is released under the MIT License.
See the `LICENSE` file for details.

The Rodeostat firmware and Python library are third-party components and remain
subject to their respective licenses.