"""Synthetic datasets with the same layout as data/v1, so every check can be tested
on data where we know the right answer (including deliberately broken copies)."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from csizer.dataset import CONFIG_FILES, PART_FILES, RUN_COLUMNS, DataPaths

LOG_TEMPLATE = """httperf: warning: open file limit > FD_SETSIZE; limiting max. # of open files to FD_SETSIZE
httperf --timeout=120 --client=0/1 --server=192.0.2.1 --port=80 --uri=/server.php?size={size}&cpuLoad={cpuLoad} --rate={arr} --send-buffer=4096 --recv-buffer=16384 --num-conns={conns} --num-calls=1
Maximum connect burst length: 1

Total: connections {conns} requests {replies} replies {replies} test-duration 15.000 s

Connection rate: {arr}.0 conn/s (100.0 ms/conn, <={conc} concurrent connections)
Connection time [ms]: min 1.0 avg {rt} max {rtmax} median {rt} stddev {rtstd}
Connection time [ms]: connect 0.5
Connection length [replies/conn]: 1.000

Request rate: {arr}.0 req/s (100.0 ms/req)
Request size [B]: 100.0

Reply rate [replies/s]: min {thr} avg {thr} max {thr} stddev 0.0 (3 samples)
Reply time [ms]: response {rt} transfer 0.0
Reply size [B]: header 174.0 content 67.0 footer 0.0 (total 241.0)
Reply status: 1xx=0 2xx={replies} 3xx=0 4xx=0 5xx=0

CPU time [s]: user 2.75 system 12.18 (user 18.4% system 81.5% total 100.0%)
Net I/O: 3.4 KB/s (0.0*10^6 bps)

Errors: total {errors} client-timo 0 socket-timo 0 connrefused {refused} connreset {reset}
Errors: fd-unavail 0 addrunavail 0 ftab-full 0 other 0
"""

PREFIX = {"train": "train", "validation": "validation", "grid_slice": "grid_slice"}


@dataclass
class Rep:
    rt: float
    thr: float | None = None  # default: equal to arr
    errors: int = 0
    refused: int = 0
    reset: int = 0


@dataclass
class Cfg:
    kind: str
    arr: int = 10
    cpus: float = 1.0
    memory: int = 512
    device_write: int = 20
    size: int = 50
    cpuLoad: int = 500_000  # noqa: N815
    reps: tuple = (Rep(16.0), Rep(16.1), Rep(16.2))
    slice_param: str = ""


def write_dataset(root: Path, cfgs: list[Cfg]) -> DataPaths:
    for sub in ("runs", "configs", "raw_logs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    rows = {k: [] for k in PART_FILES}
    plans = {k: [] for k in PART_FILES}
    counters = {k: 0 for k in PART_FILES}
    for c in cfgs:
        counters[c.kind] += 1
        cid = f"{PREFIX[c.kind]}_{counters[c.kind]:04d}"
        conns = max(50, min(800, c.arr * 15))
        plan = [c.arr, c.cpus, f"{c.memory}m", f"{c.device_write}mb", c.size, c.cpuLoad]
        plans[c.kind].append(plan + ([c.slice_param] if c.kind == "grid_slice" else []))
        for i, r in enumerate(c.reps, start=1):
            thr = c.arr if r.thr is None else r.thr
            replies = 0 if thr == 0 else conns - r.errors
            rows[c.kind].append([cid, i, *plan, c.slice_param, float(thr), r.rt, r.rt * 1.2, 0.3, r.errors, conns,
                                 c.kind])
            (root / "raw_logs" / f"{cid}_rep{i}.txt").write_text(LOG_TEMPLATE.format(
                size=c.size, cpuLoad=c.cpuLoad, arr=c.arr, conns=conns, replies=replies, conc=1, rt=r.rt,
                rtmax=r.rt * 1.2, rtstd=0.3, thr=float(thr), errors=r.errors, refused=r.refused, reset=r.reset))

    def csv(rows_):
        return "\n".join(",".join("" if v is None else str(v) for v in row) for row in rows_) + "\n"

    header = ",".join(RUN_COLUMNS)
    allrows = []
    for k, fname in PART_FILES.items():
        (root / "runs" / fname).write_text(header + "\n" + csv(rows[k]))
        allrows += rows[k]
    (root / "runs" / "raw_results.csv").write_text(header + "\n" + csv(allrows))
    for k, fname in CONFIG_FILES.items():
        h = "arr,cpus,memory,device_write,size,cpuLoad" + (",slice_param" if k == "grid_slice" else "")
        (root / "configs" / fname).write_text(h + "\n" + csv(plans[k]))
    write_manifest(root)
    return DataPaths(root)


def write_manifest(root: Path) -> None:
    lines = []
    for sub in ("runs", "configs", "raw_logs"):
        for f in sorted((root / sub).iterdir()):
            lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {sub}/{f.name}")
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")


def standard_configs() -> list[Cfg]:
    """A small, healthy dataset: two train configs, one validation config, one grid slice."""
    return [
        Cfg("train", arr=1, cpuLoad=100_000, reps=(Rep(5.6), Rep(5.7), Rep(5.8))),
        Cfg("train", arr=10, size=200, reps=(Rep(25.0), Rep(25.2), Rep(24.9))),
        Cfg("validation", arr=5, cpus=0.5, reps=(Rep(16.4), Rep(16.5), Rep(16.3))),
        Cfg("grid_slice", slice_param="arr"),
    ]


@pytest.fixture
def synth(tmp_path) -> DataPaths:
    return write_dataset(tmp_path / "v1", standard_configs())


REAL_ROOT = Path(os.environ.get("CSIZER_DATA", Path(__file__).resolve().parents[1] / "data" / "v1"))


@pytest.fixture(scope="session")
def real_paths() -> DataPaths:
    if not (REAL_ROOT / "MANIFEST.sha256").exists():
        pytest.skip("real dataset not imported")
    return DataPaths(REAL_ROOT)
