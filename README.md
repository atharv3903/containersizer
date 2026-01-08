# ContainerSizer

Empirical performance modeling and SLA-based resource sizing for a Dockerized web server.
The benchmark covers 325 CPU, memory, disk-write, and workload configurations with three httperf repeats each.

The original benchmark measurements were collected by the IIT Bombay CS681 course team.
The archived logs use the documentation address `192.0.2.1` in place of the lab endpoint.
The checksum manifest describes the published files, including the redacted logs.

This repository reconstructs the January–May 2026 course timeline from saved development snapshots.

Run the checksum check from the repository root.

```sh
cd data/v1
sha256sum --check MANIFEST.sha256
```
