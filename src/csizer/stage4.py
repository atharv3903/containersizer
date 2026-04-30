"""`csizer sizer`: offline evaluation of the sizer + example recommendations; writes the stage 4 report."""
# ruff: noqa: E501  (report-building f-strings)

from __future__ import annotations

import json
from pathlib import Path

from . import dataset as ds
from . import sizer as sz

EXAMPLES = [  # (arr, size, cpuLoad, SLA ms)
    (10, 50, 5_000_000, 300),  # long CPU requests: the measured 1-CPU container collapsed (19.6 s)
    (30, 200, 100_000, 50),  # heavy writes: 60 MB/s offered
    (20, 100, 1_000_000, 50),
    (5, 10, 500_000, 25),
    (50, 10, 100_000, 10),
]


def run(data_root: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    c = ds.build_config_table(ds.load_runs(ds.DataPaths(data_root)))
    a = sz.oof_assessments(c)
    dec = sz.decision_table(a)
    wc = sz.workload_choices(a)
    a.to_csv(out / "sizer_oof_assessments.csv", index=False)
    dec.to_csv(out / "sizer_decisions.csv", index=False)
    wc.to_csv(out / "sizer_workload_choices.csv", index=False)

    tot = dec.groupby("method")[["accepted", "bad_accepts", "needless_rejects", "truly_ok"]].sum()
    summ = {}
    for m, g in wc.groupby("method"):
        n = len(g)
        summ[m] = {"cases": n, "meets_sla": int((g.outcome == "meets_sla").sum()),
                   "violates_sla": int((g.outcome == "violates_sla").sum()), "none": int((g.outcome == "none").sum()),
                   "optimal": int(g.optimal.eq(True).sum()),
                   "mean_regret_when_met": float(g.regret.mean())}

    full = sz.fit_sizer(c[c.experiment_type == "train"])
    ex_lines = []
    for arr, size, cl, sla in EXAMPLES:
        best, table = full.recommend(arr, size, cl, sla)
        rec = (f"{best.cpus:g} CPU, {best.memory} MiB, {best.device_write} MB/s (predicted {best.pred_rt_ms:.1f} ms, "
               f"bound {best.bound_rt_ms:.1f} ms, margin {best.margin:.2f})") if best is not None else "no feasible container"
        meas = c[(c.arr == arr) & (c.mb_written == sz.mb_written_per_request(size)) & (c.cpuLoad == cl)]
        mtxt = "; ".join(f"{r.cpus:g} CPU/{r.memory}/{r.device_write}: {r.rt_ms:.0f} ms {r.label}"
                         for r in meas.itertuples()) or "none measured"
        ex_lines.append(f"| {arr} req/s, size {size}, cpuLoad {cl:,} | {sla} ms | {rec} | {mtxt} |")
    caps = []
    for cpus in (0.5, 1.0, 2.0):
        caps.append(f"| {cpus:g} | " + " | ".join(str(full.capacity(cpus, 512, 20, 50, cl, 50))
                                                  for cl in (100_000, 500_000, 1_000_000)) + " |")

    res = {"decisions_total": tot.to_dict(orient="index"), "workload_summary": summ, "guard": full.guard,
           "theta_io": full.gate.theta_io}
    (out / "sizer_results.json").write_text(json.dumps(res, indent=1, default=float))

    def pct(x, n):
        return f"{x} ({x / n:.1%})" if n else "0"

    lines = [
        "# Stage 4 sizer report (generated)", "",
        "Every prediction below is out of fold: grouped, stratified 5-fold CV over all 325 measured configurations "
        "(gate + random forest refitted per fold). Ground truth = the measurement (steady and mean response time "
        f"≤ SLA). SLAs tested: {', '.join(map(str, sz.SLAS_MS))} ms. Guard band (final model) = "
        f"+{full.guard:.1%} (P90 out-of-fold relative error).", "",
        "## 1. Per-configuration decisions (325 configurations × 7 SLAs = 2,275 decisions)", "",
        "| Method | accepted | bad accepts (would break the SLA) | needless rejects | truly OK |",
        "|---|---|---|---|---|",
        *[f"| {m} | {r.accepted} | {pct(r.bad_accepts, r.accepted)} | {pct(r.needless_rejects, r.truly_ok)} | "
          f"{r.truly_ok} |" for m, r in tot.iterrows()], "",
        "Per SLA:", "", "| SLA ms | method | accepted | bad accepts | needless rejects | truly OK |",
        "|---|---|---|---|---|---|",
        *[f"| {r.sla_ms} | {r.method} | {r.accepted} | {r.bad_accepts} | {r.needless_rejects} | {r.truly_ok} |"
          for r in dec.itertuples()], "",
        "## 2. Sizing decisions on measured workloads", "",
        "For each workload measured under 2+ containers and each SLA that at least one measured container meets: "
        "the method picks the cheapest container it believes meets the SLA, and the pick is checked against its "
        "measurement.", "",
        "| Method | cases | pick meets SLA | pick violates SLA | no pick | cheapest correct pick | mean extra cost when met |",
        "|---|---|---|---|---|---|---|",
        *[f"| {m} | {s['cases']} | {pct(s['meets_sla'], s['cases'])} | {pct(s['violates_sla'], s['cases'])} | "
          f"{s['none']} | {pct(s['optimal'], s['cases'])} | {s['mean_regret_when_met']:.1%} |" for m, s in summ.items()],
        "",
        "## 3. Example recommendations (sizer fitted on the 250 training configurations)", "",
        "| Workload | SLA | Recommendation | Measured containers for this workload |", "|---|---|---|---|",
        *ex_lines, "",
        "## 4. Capacity planning: max req/s within a 50 ms SLA (512 MiB, 20 MB/s, no writes)", "",
        "| cpus | cpuLoad 100k | 500k | 1M |", "|---|---|---|---|", *caps,
    ]
    (out / "sizer_report.md").write_text("\n".join(lines) + "\n")
    return res
