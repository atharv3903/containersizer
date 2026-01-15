"""The experiment's parameter space and the physical meaning of each parameter.

Six inputs describe one load test: three describe the *workload* (how many requests
per second, how much CPU work and how much disk work each request does) and three
describe the *container* (Docker resource limits).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

WORKLOAD_PARAMS = ("arr", "size", "cpuLoad")
RESOURCE_PARAMS = ("cpus", "memory", "device_write")
PARAMS = ("arr", "cpus", "memory", "device_write", "size", "cpuLoad")  # column order in the raw files

LEVELS: dict[str, tuple] = {
    "arr": (1, 2, 5, 10, 20, 30, 40, 50),  # requests per second offered by httperf
    "cpus": (0.5, 1.0, 2.0),  # docker --cpus (CFS quota, in CPUs)
    "memory": (512, 1024),  # docker -m, MiB
    "device_write": (10, 20, 50),  # docker --device-write-bps, MB/s
    "size": (10, 25, 50, 100, 200),  # server.php ?size= ; MB written = floor(size/100) x 1 MB
    "cpuLoad": (100_000, 500_000, 1_000_000, 5_000_000),  # server.php ?cpuLoad= ; rand() calls
}

# The file each request appends to disk is exactly 1 MiB (data.txt, 1,048,576 bytes).
MB_PER_WRITE = 1.0


@dataclass(frozen=True)
class Config:
    arr: int
    cpus: float
    memory: int
    device_write: int
    size: int
    cpuLoad: int  # noqa: N815 - keep the experiment's own name

    def key(self) -> tuple:
        return (self.arr, self.cpus, self.memory, self.device_write, self.size, self.cpuLoad)


def parse_memory(v: str | int) -> int:
    """'512m' -> 512 (MiB). Integers pass through."""
    if isinstance(v, int | float) and not isinstance(v, bool):
        return int(v)
    m = re.fullmatch(r"\s*(\d+)\s*[mM]\s*", str(v))
    if not m:
        raise ValueError(f"bad memory value {v!r}")
    return int(m.group(1))


def parse_mbps(v: str | int) -> int:
    """'10mb' -> 10 (MB/s). Integers pass through."""
    if isinstance(v, int | float) and not isinstance(v, bool):
        return int(v)
    m = re.fullmatch(r"\s*(\d+)\s*[mM][bB]\s*", str(v))
    if not m:
        raise ValueError(f"bad device_write value {v!r}")
    return int(m.group(1))


def writes_per_request(size: int) -> int:
    """How many 1 MB writes server.php really performs.

    The PHP loop is `for ($i = 1; $i <= $size/100; $i++) fwrite(...)`. For size 10, 25 and 50
    the bound is 0.1, 0.25 and 0.5, so the loop body never runs: those requests write nothing.
    """
    if size < 0:
        raise ValueError("size must be >= 0")
    return math.floor(size / 100)


def mb_written_per_request(size: int) -> float:
    return writes_per_request(size) * MB_PER_WRITE


def nominal_mb_per_request(size: int) -> float:
    """What the parameter was *meant* to write (size/100 MB). Kept to show the gap."""
    return size / 100.0


def is_on_grid(param: str, value) -> bool:
    return value in LEVELS[param]
