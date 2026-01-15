"""Loading the measured runs and building the per-configuration table.

Two levels of data:

* **runs**: one row per httperf execution (325 configurations x 3 repeats = 975 rows).
* **configs**: one row per configuration, aggregating its repeats (325 rows).

The raw files are read-only and checked against a SHA-256 manifest before use.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import httperf
from .params import PARAMS, mb_written_per_request, parse_mbps, parse_memory, writes_per_request

RUN_COLUMNS = (
    "config_id",
    "repeat",
    "arr",
    "cpus",
    "memory",
    "device_write",
    "size",
    "cpuLoad",
    "slice_param",
    "throughput_rps",
    "reply_time_mean_ms",
    "reply_time_max_ms",
    "reply_time_stddev_ms",
    "errors",
    "conns_attempted",
    "experiment_type",
)
EXPERIMENT_TYPES = ("train", "validation", "grid_slice")
PART_FILES = {"train": "raw_results_train.csv", "validation": "raw_results_validation.csv",
              "grid_slice": "raw_results_grid_slices.csv"}
CONFIG_FILES = {"train": "configs_train.csv", "validation": "configs_validation.csv",
                "grid_slice": "configs_grid_slices.csv"}

# Stability thresholds (decided in stage 1, see the generated dataset report)
STEADY_RATIO = 0.95  # completed/offered throughput at or above this ...
OVERLOAD_RATIO = 0.80  # ... below this the server is not keeping up
OVERLOAD_ERROR_RATE = 0.05
# A repeat is a "stall" if it is this many times slower than the config's fastest repeat
# and at least STALL_MIN_EXTRA_MS slower in absolute terms.
STALL_FACTOR = 3.0
STALL_MIN_EXTRA_MS = 50.0


def default_data_root() -> Path:
    env = os.environ.get("CSIZER_DATA")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data" / "v1"


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @property
    def runs(self) -> Path:
        return self.root / "runs" / "raw_results.csv"

    def part(self, kind: str) -> Path:
        return self.root / "runs" / PART_FILES[kind]

    def configs(self, kind: str) -> Path:
        return self.root / "configs" / CONFIG_FILES[kind]

    @property
    def demo(self) -> Path:
        return self.root / "runs" / "demo_live_results.csv"

    @property
    def raw_logs(self) -> Path:
        return self.root / "raw_logs"

    @property
    def manifest(self) -> Path:
        return self.root / "MANIFEST.sha256"

    def log(self, config_id: str, repeat: int) -> Path:
        return self.raw_logs / f"{config_id}_rep{repeat}.txt"


def verify_manifest(paths: DataPaths) -> list[str]:
    """Return a list of problems (empty if every file matches its recorded SHA-256)."""
    problems: list[str] = []
    listed: set[str] = set()
    for line in paths.manifest.read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        rel = rel.strip()
        listed.add(rel)
        f = paths.root / rel
        if not f.exists():
            problems.append(f"missing: {rel}")
            continue
        if hashlib.sha256(f.read_bytes()).hexdigest() != digest:
            problems.append(f"hash mismatch: {rel}")
    for sub in ("runs", "configs", "raw_logs"):
        for f in sorted((paths.root / sub).iterdir()):
            rel = f"{sub}/{f.name}"
            if rel not in listed:
                problems.append(f"not in manifest: {rel}")
    return problems


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["memory"] = df["memory"].map(parse_memory).astype(int)
    df["device_write"] = df["device_write"].map(parse_mbps).astype(int)
    df["arr"] = df["arr"].astype(int)
    df["cpus"] = df["cpus"].astype(float)
    df["size"] = df["size"].astype(int)
    df["cpuLoad"] = df["cpuLoad"].astype(int)
    return df


def read_runs_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"memory": str, "device_write": str, "slice_param": str})
    if tuple(df.columns) != RUN_COLUMNS:
        raise ValueError(f"{path}: unexpected columns {list(df.columns)}")
    return _normalise(df)


def load_runs(paths: DataPaths) -> pd.DataFrame:
    """All 975 runs, with parameters normalised to numbers and derived write columns."""
    df = read_runs_csv(paths.runs)
    df["writes_per_request"] = df["size"].map(writes_per_request)
    df["mb_written"] = df["size"].map(mb_written_per_request)
    df["failed"] = (df["reply_time_mean_ms"] <= 0) | (df["throughput_rps"] <= 0)
    df["error_rate"] = df["errors"] / df["conns_attempted"]
    return df


def load_configs_file(paths: DataPaths, kind: str) -> pd.DataFrame:
    df = pd.read_csv(paths.configs(kind), dtype={"memory": str, "device_write": str})
    return _normalise(df)


def load_logs(paths: DataPaths, runs: pd.DataFrame) -> pd.DataFrame:
    """Parse the raw httperf log behind every run (columns prefixed with log_)."""
    rows = []
    for cid, rep in zip(runs["config_id"], runs["repeat"], strict=True):
        rec = httperf.parse_file(paths.log(cid, int(rep))).as_dict()
        rows.append({"config_id": cid, "repeat": int(rep), **{f"log_{k}": v for k, v in rec.items()}})
    return pd.DataFrame(rows)


def _stall_mask(rt: pd.Series, groups: pd.Series) -> pd.Series:
    fastest = rt.where(rt > 0).groupby(groups).transform("min")
    return (rt > STALL_FACTOR * fastest) & (rt - fastest > STALL_MIN_EXTRA_MS)


def build_config_table(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per configuration.

    Response time and throughput are the median over *valid* repeats (a repeat where the
    server answered nothing is excluded from the median but counted in `n_failed`).
    """
    r = runs.copy()
    r["stall"] = _stall_mask(r["reply_time_mean_ms"], r["config_id"]) & ~r["failed"]
    valid = r[~r["failed"]]
    g_all = r.groupby("config_id", sort=False)
    g_ok = valid.groupby("config_id", sort=False)
    first = g_all.first()
    out = first[list(PARAMS) + ["experiment_type", "slice_param", "writes_per_request", "mb_written",
                                "conns_attempted"]].copy()
    out["n_repeats"] = g_all.size()
    out["n_failed"] = g_all["failed"].sum().astype(int)
    out["n_stalls"] = g_all["stall"].sum().astype(int)
    out["rt_ms"] = g_ok["reply_time_mean_ms"].median()
    out["rt_min_ms"] = g_ok["reply_time_mean_ms"].min()
    out["rt_max_ms"] = g_ok["reply_time_mean_ms"].max()
    out["rt_spread"] = out["rt_max_ms"] / out["rt_min_ms"]
    out["thr_rps"] = g_ok["throughput_rps"].median()
    out["max_conn_time_ms"] = g_ok["reply_time_max_ms"].median()
    out["errors_total"] = g_all["errors"].sum().astype(int)
    out["error_rate"] = out["errors_total"] / (out["conns_attempted"] * out["n_repeats"])
    out["thr_ratio"] = out["thr_rps"] / out["arr"]
    out["label"] = stability_label(out)
    out["config_order"] = out.index.str.extract(r"_(\d+)$", expand=False).astype(int).values
    return out.reset_index()


def stability_label(t: pd.DataFrame) -> pd.Series:
    """steady / near_saturation / overload from completed throughput and errors.

    overload: completed throughput < 80% of offered, or > 5% errors, or a repeat where the
              server answered nothing.
    near_saturation: 80-95% of offered, or any error at all.
    steady: everything else.
    """
    ratio = t["thr_ratio"].fillna(0.0)
    over = (ratio < OVERLOAD_RATIO) | (t["error_rate"] > OVERLOAD_ERROR_RATE) | (t["n_failed"] > 0)
    near = ~over & ((ratio < STEADY_RATIO) | (t["errors_total"] > 0))
    return pd.Series(np.select([over, near], ["overload", "near_saturation"], "steady"), index=t.index)


def parameter_duplicates(configs: pd.DataFrame) -> pd.DataFrame:
    """Parameter tuples measured under more than one config_id."""
    grp = configs.groupby(list(PARAMS))["config_id"].agg(list)
    grp = grp[grp.map(len) > 1]
    return grp.reset_index().rename(columns={"config_id": "config_ids"})


def validation_in_train(configs: pd.DataFrame) -> list[str]:
    """Validation configurations whose exact parameter tuple was also a training configuration."""
    train = set(map(tuple, configs.loc[configs.experiment_type == "train", list(PARAMS)].values))
    val = configs[configs.experiment_type == "validation"]
    return [cid for cid, key in zip(val["config_id"], map(tuple, val[list(PARAMS)].values), strict=True)
            if key in train]
