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