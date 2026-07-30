from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from potentiostat import Potentiostat

CSV_PATH = Path("input/conditions/electrodeposition_conditions_96cond_1.csv")
TEST_VOLT = 0.0
TEST_TIME_S = 1
WAIT_BETWEEN_PORTS_S = 0

df = pd.read_csv(CSV_PATH)
ports = df["port"].dropna().astype(str).str.strip().tolist()

param = {
    "quietValue": 0.0,
    "quietTime": 0,
    "value": TEST_VOLT,
    "duration": int(TEST_TIME_S * 1000),
}

potentiostats = [Potentiostat(port) for port in ports]

try:
    while True:
        for pst in potentiostats:
            print(pst)
            pst.run_test("constant", param=param)
            time.sleep(WAIT_BETWEEN_PORTS_S)
finally:
    for pst in potentiostats:
        try:
            pst.close()
        except Exception:
            pass
