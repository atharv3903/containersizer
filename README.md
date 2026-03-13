# ContainerSizer

Empirical performance modeling and SLA-based resource sizing for a Dockerized web server.
The benchmark covers 325 CPU, memory, disk-write, and workload configurations with three httperf repeats each.

The original benchmark measurements were collected by the IIT Bombay CS681 course team.
The archived logs use the documentation address `192.0.2.1` in place of the lab endpoint.
The checksum manifest describes the published files, including the redacted logs.

This repository reconstructs the January–May 2026 course timeline from saved development snapshots.

## Setup

Use Python 3.12 and run these commands from the repository root.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
scripts/csizer verify
```

## Results

- [Dataset audit](evidence/dataset_report.md) checks all 975 measured runs against the httperf logs.
- [Model evaluation](evidence/models_report.md) compares five regressors using grouped cross-validation.
  The random forest reaches an R² of 0.997 and an RMSE of 2.7 ms on 41 clean steady-state holdout configurations.

## Reproduce

`scripts/measure_stage2.sh` writes reports and figures to `artifacts/stage-2/`.
The `stage-2` tag marks this milestone.
