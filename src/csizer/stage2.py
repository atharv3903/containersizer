"""`csizer models`: train the five regressors, evaluate, write the stage 2 report and plots."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import dataset as ds  # noqa: E402
from . import models  # noqa: E402

ORDER = ["svr", "xgboost", "random_forest", "mlp", "ridge"]


def _fmt(df: pd.DataFrame, set_name: str) -> list[str]:
    d = df[df.set == set_name].set_index("model").loc[ORDER]
    lines = ["| Model | n | R² | RMSE (ms) | MAPE | P90-APE | R² (log) |", "|---|---|---|---|---|---|---|"]
    for m, r in d.iterrows():
        ci = ""
        if "r2_ci" in d.columns and isinstance(r.get("r2_ci"), list):
            ci = f" [{r.r2_ci[0]:.3f}, {r.r2_ci[1]:.3f}]"
        rm = f" [{r.rmse_ci[0]:.2f}, {r.rmse_ci[1]:.2f}]" if ci else ""
        p9 = f" [{r.p90_ape_ci[0]:.1%}, {r.p90_ape_ci[1]:.1%}]" if ci else ""
        lines.append(f"| {m} | {r.n} | {r.r2:.3f}{ci} | {r.rmse_ms:.2f}{rm} | {r.mape:.1%} | {r.p90_ape:.1%}{p9} "
                     f"| {r.r2_log:.3f} |")
    return lines


def run(data_root: Path, out: Path, n_iter: int = 40) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    configs = ds.build_config_table(ds.load_runs(ds.DataPaths(data_root)))
    leaked = ds.validation_in_train(configs)
    split = models.make_split(configs, leaked, drop_stalled_median=True)

    t0 = time.time()
    fitted = models.fit_all(split.train, n_iter=n_iter)
    fit_s = time.time() - t0
    res = models.evaluate(fitted, split)

    # sensitivity: keep training configs whose median is a stall
    split_ns = models.make_split(configs, leaked)
    fitted_ns = models.fit_all(split_ns.train, n_iter=n_iter)
    res_ns = models.evaluate(fitted_ns, split_ns, ci=False)

    res.to_json(out / "models_results.json", orient="records", indent=1)
    preds = split.test_all_clean[["config_id", "label", "rt_ms"]].copy()
    for n, f in fitted.items():
        preds[f"pred_{n}"] = f.predict_ms(split.test_all_clean)
    preds.round(3).to_csv(out / "validation_predictions.csv", index=False)

    clean = res[res.set == "val_clean_steady"].set_index("model")
    best = clean["p90_ape"].idxmin()

    # plot: predicted vs measured, best model, clean validation (steady + non-steady)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for lab, col in (("steady", "#2a7ab9"), ("near_saturation", "#e0a030"), ("overload", "#c8403a")):
        s = preds[preds.label == lab]
        ax.scatter(s.rt_ms, s[f"pred_{best}"], color=col, label=lab, s=25)
    lim = [3, preds.rt_ms.max() * 1.5]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="measured mean response time (ms)",
           ylabel=f"predicted ({best})", title="Validation (clean): accurate in steady state, blind to overload")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "plots" / "pred_vs_measured.png", dpi=120)
    plt.close(fig)

    lines = [
        "# Stage 2 model report (generated)", "",
        f"Training: {len(split.train)} steady training configurations, excluding configs whose median repeat is a "
        "stall (grouped 5-fold CV by parameter tuple, "
        f"{n_iter} random-search candidates per model, target log(rt)). Fit time {fit_s:.0f} s.", "",
        f"Leaked validation configs excluded from the clean sets: {', '.join(leaked)}", "",
        f"## Clean steady validation (n = {len(split.test_clean)}), 95% bootstrap CIs", "",
        *_fmt(res, "val_clean_steady"), "",
        f"## Same metrics on all {len(split.test_v1_style)} steady validation configs "
        "(includes the 5 leaked ones; the original evaluation's set)", "",
        *_fmt(res, "val_v1_style_steady"), "",
        f"## Clean validation, every label (n = {len(split.test_all_clean)}, includes "
        f"{int((split.test_all_clean.label != 'steady').sum())} non-steady)", "",
        *_fmt(res, "val_clean_all"), "",
        f"## Grouped CV out-of-fold on training (n = {len(split.train)})", "",
        *_fmt(res, "train_oof"), "",
        f"## Sensitivity: training WITH the {len(split_ns.train) - len(split.train)} config(s) whose median is a "
        "stall (clean steady validation)", "",
        *_fmt(res_ns, "val_clean_steady"), "",
        f"Best by P90-APE on clean steady validation: **{best}**", "",
        "## Best hyperparameters", "",
    ]
    for n in ORDER:
        lines.append(f"- {n}: `{json.dumps({k: (v if not isinstance(v, float) else round(v, 5)) for k, v in fitted[n].best_params.items()}, default=str)}`")  # noqa: E501
    (out / "models_report.md").write_text("\n".join(lines) + "\n")
    return {"best": best, "results": res}


def feature_importance(data_root: Path, out: Path) -> Path:
    """Permutation importance of the best tree model on clean steady validation (stage 2 plot)."""
    from sklearn.inspection import permutation_importance

    configs = ds.build_config_table(ds.load_runs(ds.DataPaths(data_root)))
    split = models.make_split(configs, ds.validation_in_train(configs), drop_stalled_median=True)
    f = models.fit_all(split.train, n_iter=20, names=["random_forest"])["random_forest"]
    X, y = models.features(split.test_clean), models.target(split.test_clean)
    pi = permutation_importance(f.estimator, X, y, n_repeats=30, random_state=0, scoring="neg_root_mean_squared_error")
    order = np.argsort(pi.importances_mean)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(np.array(models.FEATURES)[order], pi.importances_mean[order], xerr=pi.importances_std[order])
    ax.set_xlabel("increase in RMSE of log(rt) when the feature is shuffled")
    ax.set_title("Permutation importance (random forest)")
    fig.tight_layout()
    p = out / "plots" / "feature_importance.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p
