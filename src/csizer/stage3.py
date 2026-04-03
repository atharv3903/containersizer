"""`csizer stability`: fit the gate and classifiers, evaluate on held-out and grouped-CV, write the report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import dataset as ds  # noqa: E402
from . import stability as st  # noqa: E402


def _row(name: str, m: dict) -> str:
    return (f"| {name} | {m['tp']} | {m['fn']} | {m['fp']} | {m['tn']} | {m['recall']:.0%} | "
            f"{m['false_alarm_rate']:.1%} | {m['precision']:.0%} |")


HDR = ["| Method | caught (TP) | missed (FN) | false alarms (FP) | correct safe (TN) | recall | false-alarm rate "
       "| precision |", "|---|---|---|---|---|---|---|---|"]


def run(data_root: Path, out: Path) -> dict:
    (out / "plots").mkdir(parents=True, exist_ok=True)
    c = ds.build_config_table(ds.load_runs(ds.DataPaths(data_root)))
    train = c[c.experiment_type == "train"]
    hold = st.holdout_set(c)
    gate = st.fit_gate(train)
    y_tr, y_h = st.unsafe(train), st.unsafe(hold)

    res = {"calibration": gate.cal.__dict__, "theta_cpu": gate.theta_cpu, "theta_io": gate.theta_io,
           "holdout_n": len(hold), "holdout_unsafe": int(y_h.sum()), "train_unsafe": int(y_tr.sum())}
    hold_rows, hold_res = [], {}
    preds = {"gate": gate.predict_unsafe(hold)}
    for name, est in st.classifier_specs().items():
        est.fit(st.clf_features(train), y_tr)
        preds[name] = est.predict(st.clf_features(hold)).astype(bool)
    for name, p in preds.items():
        hold_res[name] = st.confusion(y_h, p)
        hold_rows.append(_row(name, hold_res[name]))
    res["holdout"] = hold_res

    cvp = st.grouped_cv_predictions(c)
    y_all = st.unsafe(c)
    cv_res = {name: st.confusion(y_all, cvp[name].to_numpy()) for name in cvp.columns}
    res["grouped_cv_all"] = cv_res

    # what the gate says about every non-steady configuration (fitted on training only)
    sc = gate.scores(c)
    ns = c[y_all].join(sc[["rho_cpu", "rho_io", "margin"]])
    res["non_steady"] = ns[["config_id", "label", "rt_ms", "thr_ratio", "rho_cpu", "rho_io", "margin"]].round(3) \
        .values.tolist()
    fa = c[~y_all].join(sc[["rho_cpu", "rho_io", "margin"]])
    fa = fa[fa.margin >= 1.0]
    res["gate_false_alarms_all"] = fa[["config_id", "rt_ms", "rho_cpu", "rho_io", "margin", "n_stalls"]].round(3) \
        .values.tolist()
    steady_all = c[~y_all]
    res["stall_rate"] = {"gate_false_alarms": float((fa.n_stalls > 0).mean()) if len(fa) else 0.0,
                         "other_steady": float((steady_all[~steady_all.config_id.isin(fa.config_id)].n_stalls > 0)
                                               .mean())}
    (out / "stability_results.json").write_text(json.dumps(res, indent=1, default=float))

    # plot: rho_cpu vs rho_io, all configs, label colours, thresholds
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    allsc = c.join(sc)
    for lab, col, mk in (("steady", "#2a7ab9", "o"), ("near_saturation", "#e0a030", "s"), ("overload", "#c8403a", "X")):
        s = allsc[allsc.label == lab]
        ax.scatter(s.rho_io + 0.01, s.rho_cpu, c=col, marker=mk, s=28 if lab == "steady" else 70, label=lab,
                   alpha=0.6 if lab == "steady" else 1)
    ax.axvline(gate.theta_io + 0.01, color="k", ls="--", lw=0.8)
    ax.axhline(gate.theta_cpu, color="k", ls="--", lw=0.8)
    ax.set(xscale="log", yscale="log", xlabel="disk utilisation ρ_io = arr × MB written / cap (+0.01)",
           ylabel="CPU utilisation ρ_cpu (calibrated)",
           title="Every non-steady configuration sits past a utilisation threshold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "plots" / "stability_map.png", dpi=120)
    plt.close(fig)

    cal = gate.cal
    lines = [
        "# Stage 3 stability report (generated)", "",
        f"CPU calibration (steady no-write training configs): demand = {cal.intercept_ms:.2f} ms + "
        f"{cal.ns_per_rand:.1f} ns × cpuLoad. Thresholds: θ_cpu = {gate.theta_cpu} (theory), "
        f"θ_io = {gate.theta_io:.2f} (lowest disk utilisation at which a training config was not steady).", "",
        f"Training: 250 configs, {res['train_unsafe']} not steady. Held-out: {len(hold)} validation + grid configs "
        f"whose tuple is not in training, {res['holdout_unsafe']} not steady.", "",
        "## Held-out configurations", "", *HDR, *hold_rows, "",
        f"## Grouped 5-fold CV over all {len(c)} configurations ({int(y_all.sum())} not steady)", "", *HDR,
        *[_row(n, m) for n, m in cv_res.items()], "",
        "## Every non-steady configuration and the gate's view (fitted on training only)", "",
        "| config | label | rt ms | completed/offered | ρ_cpu | ρ_io | margin (≥ 1 = unsafe) |",
        "|---|---|---|---|---|---|---|",
        *[f"| {r[0]} | {r[1]} | {r[2]:.0f} | {r[3]:.2f} | {r[4]:.2f} | {r[5]:.2f} | {r[6]:.2f} |"
          for r in res["non_steady"]], "",
        f"## Gate false alarms over all configurations: {len(fa)} steady configs with margin ≥ 1", "",
        f"{res['stall_rate']['gate_false_alarms']:.0%} of them had at least one stalled repeat, against "
        f"{res['stall_rate']['other_steady']:.1%} of the other steady configurations: the gate's 'false alarms' are "
        "where steady state is fragile.", "",
        "| config | rt ms | ρ_cpu | ρ_io | margin | stalled repeats |", "|---|---|---|---|---|---|",
        *[f"| {r[0]} | {r[1]:.1f} | {r[2]:.2f} | {r[3]:.2f} | {r[4]:.2f} | {int(r[5])} |"
          for r in res["gate_false_alarms_all"]],
    ]
    (out / "stability_report.md").write_text("\n".join(lines) + "\n")
    return res
