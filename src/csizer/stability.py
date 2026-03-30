"""Stage 3: will this configuration stay in steady state?

Two approaches, compared on the same held-out configurations:

* **Utilisation gate (physics).** Calibrate CPU demand per request from steady training data
  (D_cpu = a + b x cpuLoad), compute rho_cpu = arr x D_cpu / cpus and rho_io = arr x MB written / cap,
  and call a configuration unsafe when either exceeds its threshold. The CPU threshold is theory (1.0);
  the disk threshold is learned from training data, because the page cache absorbs writes.
* **Classifiers (ML).** Logistic regression and a random forest on the same configuration features,
  class-weighted because only 9 of 325 configurations are not steady.

"Unsafe" = near_saturation or overload. For sizing, a missed unsafe configuration breaks the SLA;
a false alarm only costs a larger container. Recall on unsafe configurations comes first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import models
from .params import PARAMS

SEED = 0
EPS = 1e-3  # floor for log(rho_io) when nothing is written


@dataclass(frozen=True)
class CpuCalibration:
    intercept_ms: float  # fixed per-request time (Apache + PHP start + network)
    ns_per_rand: float

    def demand_ms(self, cpuLoad) -> np.ndarray:  # noqa: N803
        return self.intercept_ms + np.asarray(cpuLoad, dtype=float) * self.ns_per_rand * 1e-6


def calibrate_cpu(train: pd.DataFrame) -> CpuCalibration:
    """Fit rt = a + b x cpuLoad on steady, no-write, unthrottled (cpus >= 1, cpuLoad <= 1M) training configs.

    At low load the response time of these requests is their service demand, so the fit is the demand.
    """
    s = train[(train.label == "steady") & (train.writes_per_request == 0) & (train.cpus >= 1.0)
              & (train.cpuLoad <= 1_000_000) & (train.n_stalls < 2)]
    b, a = np.polyfit(s["cpuLoad"].astype(float), s["rt_ms"], 1)
    return CpuCalibration(float(a), float(b * 1e6))


def utilisation(c: pd.DataFrame, cal: CpuCalibration) -> pd.DataFrame:
    u = pd.DataFrame(index=c.index)
    u["rho_cpu"] = c["arr"] * cal.demand_ms(c["cpuLoad"]) / 1000.0 / c["cpus"]
    u["rho_io"] = c["arr"] * c["mb_written"] / c["device_write"]
    # one request cannot use more than one CPU: a single request with demand > quota period also stretches
    u["single_request_cpu_s"] = cal.demand_ms(c["cpuLoad"]) / 1000.0 / np.minimum(c["cpus"], 1.0)
    return u


def unsafe(c: pd.DataFrame) -> np.ndarray:
    return (c["label"] != "steady").to_numpy()


@dataclass(frozen=True)
class Gate:
    cal: CpuCalibration
    theta_cpu: float
    theta_io: float

    def scores(self, c: pd.DataFrame) -> pd.DataFrame:
        u = utilisation(c, self.cal)
        u["cpu_margin"] = u["rho_cpu"] / self.theta_cpu
        u["io_margin"] = u["rho_io"] / self.theta_io
        u["margin"] = u[["cpu_margin", "io_margin"]].max(axis=1)  # >= 1 means unsafe
        return u

    def predict_unsafe(self, c: pd.DataFrame) -> np.ndarray:
        return (self.scores(c)["margin"] >= 1.0).to_numpy()


def fit_gate(train: pd.DataFrame, theta_cpu: float = 1.0) -> Gate:
    """theta_io = the lowest disk utilisation at which a *disk-dominated* training configuration was not steady.

    Only failures where the disk term dominates count: a CPU-bound failure has rho_io = 0 and would otherwise
    set the disk threshold to 0 (found in grouped CV, see STAGE_3_REPORT P1). Default 1.0 (theory).
    """
    cal = calibrate_cpu(train)
    u = utilisation(train, cal)
    bad = unsafe(train) & (u["rho_io"] >= u["rho_cpu"]).to_numpy()
    theta_io = float(u.loc[bad, "rho_io"].min()) if bad.any() else 1.0
    return Gate(cal, theta_cpu, theta_io)


# --- classifiers -----------------------------------------------------------------------------------


def clf_features(c: pd.DataFrame) -> pd.DataFrame:
    f = models.features(c).copy()
    f["log_rho_io"] = np.log(f["rho_io"] + EPS)
    f["log_rho_cpu"] = np.log(f["rho_cpu"])
    return f


def classifier_specs() -> dict[str, object]:
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", C=1.0,
                                                                       max_iter=5000)),
        "random_forest_clf": RandomForestClassifier(n_estimators=500, min_samples_leaf=1, class_weight="balanced",
                                                    random_state=SEED, n_jobs=-1),
    }


# --- evaluation ------------------------------------------------------------------------------------


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum(y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "false_alarm_rate": fp / (fp + tn) if fp + tn else float("nan"),
            "precision": tp / (tp + fp) if tp + fp else float("nan")}


def holdout_set(configs: pd.DataFrame) -> pd.DataFrame:
    """Validation + grid-slice configurations whose parameter tuple never appears in training."""
    train_keys = set(map(tuple, configs.loc[configs.experiment_type == "train", list(PARAMS)].values))
    h = configs[configs.experiment_type != "train"]
    keep = [tuple(k) not in train_keys for k in h[list(PARAMS)].values]
    return h[keep]


def grouped_cv_predictions(configs: pd.DataFrame, folds: int = 5) -> pd.DataFrame:
    """Out-of-fold unsafe predictions for every configuration (all 325), grouped by tuple, stratified by label."""
    y = unsafe(configs)
    g = models.groups(configs)
    out = pd.DataFrame(index=configs.index)
    out["gate"] = False
    for name in classifier_specs():
        out[name] = False
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for tr, te in cv.split(configs, y, g):
        trn, tst = configs.iloc[tr], configs.iloc[te]
        out.iloc[te, out.columns.get_loc("gate")] = fit_gate(trn).predict_unsafe(tst)
        for name, est in classifier_specs().items():
            est.fit(clf_features(trn), y[tr])
            out.iloc[te, out.columns.get_loc(name)] = est.predict(clf_features(tst)).astype(bool)
    return out
