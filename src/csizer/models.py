"""Stage 2: features, five regressors, grouped cross-validated tuning, evaluation with bootstrap CIs.

Target: log(mean response time). Models are trained on steady-state training configurations only
(overloaded response times depend on test length, not just the configuration; stage 3 handles regime).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .params import PARAMS

SEED = 0
FEATURES = (
    "log_arr", "cpus", "memory", "device_write", "mb_written", "log_cpuLoad",
    "log_cpu_work_per_cpu",  # log(cpuLoad / cpus): CPU time per request as the quota sees it
    "rho_cpu",  # arr * cpuLoad / cpus, scaled: offered CPU work per unit of quota
    "rho_io",  # arr * mb_written / device_write: offered write bandwidth / cap
)


def features(c: pd.DataFrame) -> pd.DataFrame:
    """Model inputs from the configuration table. No measured constants are used, only the parameters."""
    f = pd.DataFrame(index=c.index)
    f["log_arr"] = np.log(c["arr"])
    f["cpus"] = c["cpus"]
    f["memory"] = c["memory"]
    f["device_write"] = c["device_write"]
    f["mb_written"] = c["mb_written"]
    f["log_cpuLoad"] = np.log(c["cpuLoad"])
    f["log_cpu_work_per_cpu"] = np.log(c["cpuLoad"] / c["cpus"])
    f["rho_cpu"] = c["arr"] * c["cpuLoad"] / c["cpus"] * 1e-8
    f["rho_io"] = c["arr"] * c["mb_written"] / c["device_write"]
    return f[list(FEATURES)]


def target(c: pd.DataFrame) -> np.ndarray:
    return np.log(c["rt_ms"].to_numpy())


def groups(c: pd.DataFrame) -> np.ndarray:
    """Parameter tuple id: duplicated configurations always fall in the same CV fold."""
    return c[list(PARAMS)].astype(str).agg("|".join, axis=1).factorize()[0]


@dataclass
class Split:
    train: pd.DataFrame
    test_clean: pd.DataFrame  # steady validation configs not present in training
    test_v1_style: pd.DataFrame  # all steady validation configs (what the original evaluation used)
    test_all_clean: pd.DataFrame  # clean validation configs of every label (incl. overload)


def make_split(configs: pd.DataFrame, leaked: list[str], drop_stalled_median: bool = False) -> Split:
    c = configs
    tr = c[(c.experiment_type == "train") & (c.label == "steady")]
    if drop_stalled_median:
        tr = tr[tr.n_stalls < 2]
    val = c[c.experiment_type == "validation"]
    clean = val[~val.config_id.isin(leaked)]
    return Split(tr, clean[clean.label == "steady"], val[val.label == "steady"], clean)
