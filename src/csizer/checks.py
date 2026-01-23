"""Data-quality checks on the measured dataset.

Each check returns a `Check` with a status:
  ok   - property holds
  warn - property does not hold, but the data stays usable; the handling is documented
  fail - the dataset cannot be trusted; stop
  info - a measured fact about the data (no pass/fail)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import dataset as ds
from .params import LEVELS, PARAMS


@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail | info
    summary: str
    details: dict = field(default_factory=dict)


@dataclass
class Bundle:
    paths: ds.DataPaths
    runs: pd.DataFrame
    logs: pd.DataFrame
    configs: pd.DataFrame


def load_bundle(paths: ds.DataPaths) -> Bundle:
    runs = ds.load_runs(paths)
    logs = ds.load_logs(paths, runs)
    configs = ds.build_config_table(runs)
    return Bundle(paths, runs, logs, configs)


# --- integrity -------------------------------------------------------------------------


def check_manifest(b: Bundle) -> Check:
    problems = ds.verify_manifest(b.paths)
    if problems:
        return Check("manifest", "fail", f"{len(problems)} file(s) differ from MANIFEST.sha256",
                     {"problems": problems[:20]})
    n = len(b.paths.manifest.read_text().splitlines())
    return Check("manifest", "ok", f"all {n} files match their SHA-256", {"files": n})


def check_parts(b: Bundle) -> Check:
    parts = pd.concat([ds.read_runs_csv(b.paths.part(k)) for k in ds.EXPERIMENT_TYPES], ignore_index=True)
    base = ds.read_runs_csv(b.paths.runs)
    if not base.equals(parts):
        return Check("parts_union", "fail", "raw_results.csv is not train + validation + grid_slices")
    counts = base["experiment_type"].value_counts().to_dict()
    return Check("parts_union", "ok", f"raw_results.csv = train + validation + grid_slices ({len(base)} runs)",
                 {"runs_by_type": counts})


def check_configs_match(b: Bundle) -> Check:
    bad = {}
    for kind in ds.EXPERIMENT_TYPES:
        planned = ds.load_configs_file(b.paths, kind)[list(PARAMS)].reset_index(drop=True)
        ran = (b.configs[b.configs.experiment_type == kind].sort_values("config_order")[list(PARAMS)]
               .reset_index(drop=True))
        if len(planned) != len(ran) or not planned.equals(ran):
            bad[kind] = {"planned": len(planned), "ran": len(ran)}
    if bad:
        return Check("configs_match", "fail", "measured configurations differ from the planned files", bad)
    return Check("configs_match", "ok", "every planned configuration was run, in file order",
                 {k: int((b.configs.experiment_type == k).sum()) for k in ds.EXPERIMENT_TYPES})


def check_repeats(b: Bundle) -> Check:
    counts = b.runs.groupby("config_id").size()
    reps = b.runs.groupby("config_id")["repeat"].apply(lambda s: sorted(s.tolist()))
    wrong = counts[counts != 3].index.tolist() + [c for c, r in reps.items() if r != [1, 2, 3]]
    if wrong:
        return Check("repeats", "fail", f"{len(set(wrong))} configs without exactly repeats 1,2,3",
                     {"configs": sorted(set(wrong))[:20]})
    return Check("repeats", "ok", f"all {len(counts)} configurations have repeats 1, 2, 3")


def check_grid(b: Bundle) -> Check:
    off = {p: sorted(set(b.runs[p]) - set(LEVELS[p])) for p in PARAMS}
    off = {p: v for p, v in off.items() if v}
    if off:
        return Check("on_grid", "fail", "parameter values outside the design levels", off)
    return Check("on_grid", "ok", "every parameter value is one of the design levels")


def check_logs_agree(b: Bundle) -> Check:
    m = b.runs.merge(b.logs, on=["config_id", "repeat"], how="left", validate="one_to_one")
    pairs = {
        "reply_time_mean_ms": "log_response_ms",
        "throughput_rps": "log_reply_rate_avg",
        "reply_time_max_ms": "log_conn_time_max_ms",
        "reply_time_stddev_ms": "log_conn_time_stddev_ms",
        "errors": "log_errors_total",
        "conns_attempted": "log_num_conns",
        "arr": "log_rate",
    }
    mism = {}
    for csv_col, log_col in pairs.items():
        bad = ~np.isclose(m[csv_col].astype(float), m[log_col].astype(float))
        if bad.any():
            mism[csv_col] = m.loc[bad, ["config_id", "repeat"]].astype(str).agg("_rep".join, axis=1).tolist()[:10]
    uri_ok = all(f"size={s}&cpuLoad={c}" in u for s, c, u in zip(m["size"], m["cpuLoad"], m["log_uri"],
                                                                  strict=True))
    if mism or not uri_ok:
        return Check("logs_agree", "fail", "CSV values differ from the raw httperf logs",
                     {"mismatches": mism, "uri_ok": uri_ok})
    return Check("logs_agree", "ok",
                 f"all {len(m)} CSV rows equal their raw log on {len(pairs)} fields and the request URI",
                 {"fields": list(pairs)})


# --- quality ---------------------------------------------------------------------------


def check_failed_repeats(b: Bundle) -> Check:
    f = b.runs[b.runs.failed | (b.runs.errors > 0)]
    m = f.merge(b.logs, on=["config_id", "repeat"])
    rows = [
        {"run": f"{r.config_id}_rep{r.repeat}", "errors": int(r.errors), "replies": int(r.log_replies),
         "conn_refused": int(r.log_err_conn_refused), "conn_reset": int(r.log_err_conn_reset)}
        for r in m.itertuples()
    ]
    if not rows:
        return Check("errors", "ok", "no run had errors")
    return Check("errors", "warn",
                 f"{len(rows)} of {len(b.runs)} runs had errors ({int(b.runs.failed.sum())} answered nothing); "
                 "the configuration is labelled overload and the dead repeat is excluded from the median",
                 {"runs": rows})


def check_repeat_agreement(b: Bundle) -> Check:
    c = b.configs
    cv = b.runs[~b.runs.failed].groupby("config_id")["reply_time_mean_ms"].agg(lambda s: s.std() / s.mean())
    stalls = b.runs.assign(stall=ds._stall_mask(b.runs.reply_time_mean_ms, b.runs.config_id) & ~b.runs.failed)
    st = stalls[stalls.stall]
    affected = c[c.n_stalls > 0]
    median_hit = affected[affected.n_stalls >= 2]
    pos = st["repeat"].value_counts().sort_index().to_dict()
    writes = affected["writes_per_request"].value_counts().to_dict()
    return Check(
        "repeat_agreement",
        "warn" if len(affected) else "ok",
        f"repeats agree closely for most configs (median CV {cv.median():.1%}); {len(st)} stall repeat(s) "
        f"in {len(affected)} configs; {len(median_hit)} configs have 2+ stalls so their median is a stall",
        {
            "cv_quantiles": {str(q): round(float(v), 4) for q, v in cv.quantile([0.5, 0.9, 0.95, 0.99]).items()},
            "stall_runs": st[["config_id", "repeat", "reply_time_mean_ms"]].values.tolist(),
            "stall_repeat_position": {int(k): int(v) for k, v in pos.items()},
            "stall_configs_by_writes_per_request": {int(k): int(v) for k, v in writes.items()},
            "configs_with_2plus_stalls": median_hit["config_id"].tolist(),
        },
    )


def check_duplicates(b: Bundle) -> Check:
    dups = ds.parameter_duplicates(b.configs)
    leak = ds.validation_in_train(b.configs)
    n_unique = b.configs.groupby(list(PARAMS)).ngroups
    return Check(
        "duplicates",
        "warn" if leak else "ok",
        f"{len(b.configs)} configurations, {n_unique} distinct parameter tuples; {len(dups)} tuples measured "
        f"more than once; {len(leak)} validation configs also appear in training",
        {"validation_in_train": leak, "duplicate_tuples": [list(map(str, x)) for x in dups.values.tolist()]},
    )


def check_labels(b: Bundle) -> Check:
    by = b.configs.groupby(["experiment_type", "label"]).size().unstack(fill_value=0)
    total = b.configs["label"].value_counts().to_dict()
    over = b.configs[b.configs.label != "steady"][["config_id", "label", "thr_ratio", "rt_ms", "n_failed"]]
    return Check(
        "labels",
        "info",
        "stability labels: " + ", ".join(f"{k} {v}" for k, v in total.items()),
        {"by_type": {k: {kk: int(vv) for kk, vv in row.items()} for k, row in by.iterrows()},
         "non_steady": over.round(3).values.tolist()},
    )


# --- facts about the system under test -------------------------------------------------


def _grid(b: Bundle, param: str) -> pd.DataFrame:
    g = b.configs[(b.configs.experiment_type == "grid_slice") & (b.configs.slice_param == param)]
    return g.sort_values(param)[[param, "rt_ms", "thr_ratio"]]


def fact_zero_write_sizes(b: Bundle) -> Check:
    g = _grid(b, "size")
    rt = dict(zip(g["size"], g["rt_ms"], strict=True))
    no_write = [rt[s] for s in (10, 25, 50)]
    spread = max(no_write) - min(no_write)
    return Check(
        "zero_write_sizes",
        "info",
        f"size 10/25/50 write nothing (PHP loop bound < 1): grid-slice RT {min(no_write):.1f}-{max(no_write):.1f} ms "
        f"(spread {spread:.1f} ms); size 100 (1 MB) {rt[100]:.1f} ms; size 200 (2 MB) {rt[200]:.1f} ms",
        {"grid_rt_by_size": {int(k): float(v) for k, v in rt.items()}},
    )


def fact_flat_params(b: Bundle) -> Check:
    out = {}
    for p in ("memory", "device_write", "cpus", "arr", "cpuLoad"):
        g = _grid(b, p)
        out[p] = {str(k): float(v) for k, v in zip(g[p], g["rt_ms"], strict=True)}
    mem = list(out["memory"].values())
    return Check(
        "one_at_a_time_effects",
        "info",
        f"at the midpoint workload (arr 10, size 50, cpuLoad 500k) memory changes RT by "
        f"{abs(mem[1] - mem[0]):.1f} ms, device_write by "
        f"{max(out['device_write'].values()) - min(out['device_write'].values()):.1f} ms, cpus by "
        f"{max(out['cpus'].values()) - min(out['cpus'].values()):.1f} ms",
        {"grid_rt": out},
    )


def fact_cpu_cost(b: Bundle) -> Check:
    s = b.configs[(b.configs.label == "steady") & (b.configs.writes_per_request == 0)
                  & (b.configs.cpus >= 1.0) & (b.configs.cpuLoad <= 1_000_000)]
    slope, icpt = np.polyfit(s["cpuLoad"].astype(float), s["rt_ms"], 1)
    pred = slope * s["cpuLoad"] + icpt
    r2 = 1 - ((s["rt_ms"] - pred) ** 2).sum() / ((s["rt_ms"] - s["rt_ms"].mean()) ** 2).sum()
    return Check(
        "cpu_cost_per_rand",
        "info",
        f"one rand() call costs {slope * 1e6:.1f} ns (fit on {len(s)} steady, no-write configs with cpus >= 1 and "
        f"cpuLoad <= 1M; intercept {icpt:.2f} ms, R^2 {r2:.4f})",
        {"ns_per_rand": float(slope * 1e6), "intercept_ms": float(icpt), "r2": float(r2), "n": int(len(s))},
    )


def fact_run_order_drift(b: Bundle) -> Check:
    t = b.configs[(b.configs.experiment_type == "train") & (b.configs.label == "steady")
                  & (b.configs.writes_per_request == 0)].copy()
    slope, icpt = np.polyfit(t["cpuLoad"].astype(float), t["rt_ms"], 1)
    t["res"] = t["rt_ms"] - (slope * t["cpuLoad"] + icpt)
    t["quarter"] = pd.qcut(t["config_order"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    q = t.groupby("quarter", observed=True)["res"].median().round(2).to_dict()
    rng = max(q.values()) - min(q.values())
    return Check(
        "run_order_drift",
        "info" if rng < 1.0 else "warn",
        f"median residual by quarter of the training run order varies by {rng:.2f} ms (no visible drift if < 1 ms)",
        {"median_residual_ms_by_quarter": {str(k): float(v) for k, v in q.items()}, "n": int(len(t))},
    )


def fact_demo_reproducibility(b: Bundle) -> Check:
    if not b.paths.demo.exists():
        return Check("demo_reproducibility", "info", "no demo file")
    demo = ds.read_runs_csv(b.paths.demo)
    keys = list(PARAMS)
    m = demo.merge(b.configs[b.configs.experiment_type == "train"][keys + ["config_id", "rt_ms"]], on=keys)
    m["diff_ms"] = m["reply_time_mean_ms"] - m["rt_ms"]
    worst = float(m["diff_ms"].abs().max())
    return Check(
        "demo_reproducibility",
        "info",
        f"{len(demo)} later re-runs of {m.config_id_x.nunique()} configurations ({m.config_id_y.nunique()} matching "
        f"training config_ids) are within {worst:.1f} ms of the original medians",
        {"rows": m[["config_id_x", "config_id_y", "reply_time_mean_ms", "rt_ms", "diff_ms"]].round(2).values.tolist()},
    )


DESIGN_NS_PER_RAND = 50.0  # the cost the experiment design assumed for one rand() call
DESIGN_SOFT_RHO = 0.85  # design: configs predicted above this were meant to be near saturation


def design_rho(c: pd.DataFrame, ns_per_rand: float = DESIGN_NS_PER_RAND, real_writes: bool = False) -> pd.Series:
    """Utilisation the design formula predicts: max(CPU, disk). With real_writes=True the disk term uses the
    writes server.php really performs (floor(size/100) MB) instead of the intended size/100 MB."""
    mb = c["mb_written"] if real_writes else c["size"] / 100.0
    rho_io = c["arr"] * mb / c["device_write"]
    rho_cpu = c["arr"] * c["cpuLoad"] * ns_per_rand * 1e-9 / c["cpus"]
    return np.maximum(rho_io, rho_cpu)


def fact_design_vs_outcome(b: Bundle) -> Check:
    c = b.configs[b.configs.experiment_type != "grid_slice"].copy()
    ns = fact_cpu_cost(b).details["ns_per_rand"]
    c["rho_design"] = design_rho(c)
    c["rho_corrected"] = design_rho(c, ns_per_rand=ns, real_writes=True)
    planned_near = c["rho_design"] > DESIGN_SOFT_RHO
    measured_non_steady = c["label"] != "steady"
    corrected_high = c["rho_corrected"] > DESIGN_SOFT_RHO
    return Check(
        "design_vs_outcome",
        "info",
        f"the design predicted {int(planned_near.sum())} of {len(c)} train+validation configs near saturation "
        f"(rho > {DESIGN_SOFT_RHO}); {int(measured_non_steady.sum())} were measured non-steady. With the measured "
        f"{ns:.1f} ns per rand() and the real writes, {int(corrected_high.sum())} configs exceed {DESIGN_SOFT_RHO}",
        {
            "planned_near_saturation": int(planned_near.sum()),
            "measured_non_steady": int(measured_non_steady.sum()),
            "corrected_rho_above_soft": int(corrected_high.sum()),
            "non_steady_with_rho": c.loc[measured_non_steady, ["config_id", "label", "rho_design", "rho_corrected"]]
            .round(3).values.tolist(),
            "rho_design_quantiles": c["rho_design"].quantile([0.5, 0.9, 1.0]).round(3).tolist(),
            "rho_corrected_quantiles": c["rho_corrected"].quantile([0.5, 0.9, 1.0]).round(3).tolist(),
        },
    )


ALL_CHECKS = (
    check_manifest,
    check_parts,
    check_configs_match,
    check_repeats,
    check_grid,
    check_logs_agree,
    check_failed_repeats,
    check_repeat_agreement,
    check_duplicates,
    check_labels,
    fact_zero_write_sizes,
    fact_flat_params,
    fact_cpu_cost,
    fact_run_order_drift,
    fact_demo_reproducibility,
    fact_design_vs_outcome,
)


def run_all(b: Bundle) -> list[Check]:
    return [fn(b) for fn in ALL_CHECKS]
