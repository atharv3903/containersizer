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

The audit compares 975 CSV rows against their raw logs and identifies six validation
configurations that also occur in training. See the [dataset report](evidence/dataset_report.md).

## Reproduce

`scripts/measure_stage1.sh` writes reports and figures to `artifacts/stage-1/`.
The `stage-1` tag marks this milestone.
