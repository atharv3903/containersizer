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
from sklearn.model_selection import StratifiedGroupKFold

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


def fit_sizer(train: pd.DataFrame, n_iter: int = 40) -> Sizer:
    gate = st.fit_gate(train)
    reg_train = train[(train.label == "steady") & (train.n_stalls < 2)]
    reg = models.fit_all(reg_train, n_iter=n_iter, names=["random_forest"])["random_forest"]
    y = reg_train["rt_ms"].to_numpy()
    guard = float(np.quantile(np.abs(reg.oof_pred_ms - y) / y, 0.9))  # P90 relative error, out of fold
    return Sizer(gate, reg, guard)


# --- offline evaluation --------------------------------------------------------------------------------


def oof_assessments(configs: pd.DataFrame, folds: int = 5, n_iter: int = 15) -> pd.DataFrame:
    """For every measured configuration: gate margin and predicted response time from a sizer fitted
    without that configuration's tuple (grouped, stratified by unsafe)."""
    y = st.unsafe(configs)
    g = models.groups(configs)
    parts = []
    for tr, te in StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED).split(configs, y, g):
        s = fit_sizer(configs.iloc[tr], n_iter=n_iter)
        a = s.assess(configs.iloc[te])
        a["guard"] = s.guard
        parts.append(a)
    return pd.concat(parts).loc[configs.index]


def decision_table(a: pd.DataFrame, slas=SLAS_MS) -> pd.DataFrame:
    """Per configuration and SLA: did the sizer accept the container, and did the measurement meet the SLA?"""
    rows = []
    for sla in slas:
        truth = (a["label"] == "steady") & (a["rt_ms"] <= sla)
        sizer = a["safe"] & (a["bound_rt_ms"] <= sla)
        reg_only = a["pred_rt_ms"] <= sla  # point prediction, no gate, no guard (the naive design)
        for name, acc in (("sizer", sizer), ("regressor_only", reg_only)):
            rows.append({"sla_ms": sla, "method": name,
                         "accepted": int(acc.sum()),
                         "bad_accepts": int((acc & ~truth).sum()),  # would break the SLA in reality
                         "needless_rejects": int((~acc & truth).sum()),
                         "truly_ok": int(truth.sum())})
    return pd.DataFrame(rows)


def workload_choices(a: pd.DataFrame, slas=SLAS_MS) -> pd.DataFrame:
    """Sizing among the containers actually measured for each workload (so every choice can be checked).

    Workload = (arr, MB written per request, cpuLoad); sizes below 100 write nothing and behave identically.
    """
    a = a.copy()
    a["cost"] = cost(a.cpus, a.memory, a.device_write)
    rows = []
    for (arr, mb, cl), w in a.groupby(["arr", "mb_written", "cpuLoad"]):
        w = w.groupby(["cpus", "memory", "device_write"], as_index=False).agg(
            cost=("cost", "first"), rt_ms=("rt_ms", "median"), steady=("label", lambda s: (s == "steady").all()),
            safe=("safe", "all"), bound=("bound_rt_ms", "max"), pred=("pred_rt_ms", "max"))
        if len(w) < 2:
            continue
        w = w.sort_values(["cost", "cpus", "memory", "device_write"])
        for sla in slas:
            truth = w.steady & (w.rt_ms <= sla)
            if not truth.any():
                continue  # no measured container meets this SLA: nothing to judge
            best_true = w[truth].iloc[0]
            for name, acc in (("sizer", w.safe & (w.bound <= sla)), ("regressor_only", w.pred <= sla)):
                pick = w[acc]
                if not len(pick):
                    rows.append({"workload": f"{arr}/{mb:g}MB/{cl}", "sla_ms": sla, "method": name, "outcome": "none"})
                    continue
                p = pick.iloc[0]
                ok = bool(p.steady and p.rt_ms <= sla)
                rows.append({"workload": f"{arr}/{mb:g}MB/{cl}", "sla_ms": sla, "method": name,
                             "outcome": "meets_sla" if ok else "violates_sla",
                             "optimal": bool(ok and p.cost == best_true.cost),
                             "regret": float(p.cost / best_true.cost - 1) if ok else np.nan,
                             "measured_rt_ms": float(p.rt_ms), "n_options": len(w)})
    return pd.DataFrame(rows)
