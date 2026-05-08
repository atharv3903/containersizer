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
- [Stability evaluation](evidence/stability_report.md) compares a utilization gate with classifiers.
  The gate detects all nine unsafe configurations in grouped cross-validation with a 6.6% false-alarm rate.
- [Sizing evaluation](evidence/sizer_report.md) reports **0 SLA violations in 268 workload/SLA cases**,
  with 252 feasible picks and 16 cases without a pick. It selected the cheapest correct option in 227 cases.
  The separate per-configuration check reports two bad accepts across 2,275 configuration/SLA pairs.

## Usage

Recommend the cheapest feasible container for a workload, or estimate the arrival rate that a container can sustain.

```sh
scripts/csizer recommend --arr 10 --size 50 --cpuLoad 5000000 --sla-ms 300
scripts/csizer capacity --cpus 1 --memory-mib 512 --device-write 20 --size 50 --cpuLoad 500000 --sla-ms 50
```

The SLA applies to mean response time for workloads resembling the benchmark.

## Reproduce

`scripts/measure_stage4.sh` writes reports and figures to `artifacts/stage-4/`.
The `stage-4` tag marks this milestone.
The [stage tags](https://github.com/atharv3903/containersizer/tags) cover the dataset audit,
model comparison, stability gate, and sizing engine.
