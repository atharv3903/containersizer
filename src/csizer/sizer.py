"""Stage 4: the sizer. Cheapest container that is predicted stable AND predicted under the SLA.

    recommend(workload, SLA):
        for each of the 18 container options (cpus x memory x device_write):
            safe     = utilisation gate margin < 1            (stage 3)
            pred_rt  = random forest prediction                (stage 2)
            meets    = safe and pred_rt x (1 + guard) <= SLA   (guard = model's P90 relative error)
        return the cheapest option that meets the SLA, or "no feasible container"

Evaluation is offline: every prediction comes from models that never saw that configuration (grouped CV),
and is compared with what was actually measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from . import models
from . import stability as st
from .params import LEVELS, mb_written_per_request, writes_per_request

SEED = 0
SLAS_MS = (10, 20, 30, 50, 100, 200, 500)

# Relative prices per hour. Weights follow the usual public-cloud ratio (1 vCPU ~ 9x the price of 1 GiB of RAM);
# disk bandwidth is priced like provisioned throughput, which is cheap. Only the ordering matters to the sizer.
PRICE_VCPU = 1.0
PRICE_GIB = 0.11
PRICE_MBPS = 0.002


def cost(cpus, memory_mib, device_write) -> np.ndarray:
    return (PRICE_VCPU * np.asarray(cpus, float) + PRICE_GIB * np.asarray(memory_mib, float) / 1024
            + PRICE_MBPS * np.asarray(device_write, float))


def containers() -> pd.DataFrame:
    rows = list(product(LEVELS["cpus"], LEVELS["memory"], LEVELS["device_write"]))
    df = pd.DataFrame(rows, columns=["cpus", "memory", "device_write"])
    df["cost"] = cost(df.cpus, df.memory, df.device_write)
    return df.sort_values(["cost", "cpus", "memory", "device_write"]).reset_index(drop=True)


def workload_grid(arr: int, size: int, cpuLoad: int) -> pd.DataFrame:  # noqa: N803
    g = containers()
    g["arr"], g["size"], g["cpuLoad"] = arr, size, cpuLoad
    g["writes_per_request"] = writes_per_request(size)
    g["mb_written"] = mb_written_per_request(size)
    return g


@dataclass
class Sizer:
    gate: st.Gate
    regressor: models.Fitted
    guard: float  # relative safety margin added to predicted response time

    def assess(self, c: pd.DataFrame) -> pd.DataFrame:
        out = c.copy()
        out["margin"] = self.gate.scores(c)["margin"].to_numpy()
        out["safe"] = out["margin"] < 1.0
        out["pred_rt_ms"] = self.regressor.predict_ms(c)
        out["bound_rt_ms"] = out["pred_rt_ms"] * (1 + self.guard)
        return out

    def recommend(self, arr: int, size: int, cpuLoad: int, sla_ms: float) -> tuple[pd.Series | None, pd.DataFrame]:  # noqa: N803
        a = self.assess(workload_grid(arr, size, cpuLoad))
        a["meets"] = a["safe"] & (a["bound_rt_ms"] <= sla_ms)
        ok = a[a.meets]
        return (ok.iloc[0] if len(ok) else None), a

    def capacity(self, cpus: float, memory: int, device_write: int, size: int, cpuLoad: int,  # noqa: N803
                 sla_ms: float, max_arr: int = 50) -> int:
        """Highest integer arrival rate (1..max_arr) the container serves within the SLA; 0 if none."""
        c = pd.DataFrame({"arr": np.arange(1, max_arr + 1)})
        c["cpus"], c["memory"], c["device_write"] = cpus, memory, device_write
        c["size"], c["cpuLoad"] = size, cpuLoad
        c["writes_per_request"], c["mb_written"] = writes_per_request(size), mb_written_per_request(size)
        a = self.assess(c)
        ok = (a["safe"] & (a["bound_rt_ms"] <= sla_ms)).to_numpy()
        # capacity = last rate before the first failure (response time grows with load)
        return int(np.argmin(ok)) if not ok.all() else max_arr
