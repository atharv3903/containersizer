"""Stage 2: features, five regressors, grouped cross-validated tuning, evaluation with bootstrap CIs.

Target: log(mean response time). Models are trained on steady-state training configurations only
(overloaded response times depend on test length, not just the configuration; stage 3 handles regime).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

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


def model_specs() -> dict[str, tuple]:
    """name -> (estimator, search space). Scale-sensitive models get a StandardScaler."""
    return {
        "ridge": (make_pipeline(StandardScaler(), Ridge()),
                  {"ridge__alpha": loguniform(1e-4, 1e3)}),
        "svr": (make_pipeline(StandardScaler(), SVR(kernel="rbf")),
                {"svr__C": loguniform(1e-1, 1e3), "svr__epsilon": loguniform(1e-3, 0.3),
                 "svr__gamma": loguniform(1e-3, 1)}),
        "random_forest": (RandomForestRegressor(random_state=SEED, n_jobs=1),
                          {"n_estimators": randint(100, 600), "max_depth": [None, 4, 6, 8, 12],
                           "min_samples_leaf": randint(1, 6), "max_features": [1.0, 0.7, 0.5, "sqrt"]}),
        "xgboost": (XGBRegressor(random_state=SEED, n_jobs=1, tree_method="hist"),
                    {"n_estimators": randint(100, 800), "learning_rate": loguniform(0.01, 0.3),
                     "max_depth": randint(2, 7), "subsample": uniform(0.6, 0.4),
                     "colsample_bytree": uniform(0.6, 0.4), "reg_lambda": loguniform(1e-3, 10),
                     "min_child_weight": randint(1, 6)}),
        "mlp": (make_pipeline(StandardScaler(), MLPRegressor(max_iter=5000, random_state=SEED, early_stopping=False)),
                {"mlpregressor__hidden_layer_sizes": [(16,), (32,), (64,), (32, 16), (64, 32)],
                 "mlpregressor__alpha": loguniform(1e-5, 1e-1),
                 "mlpregressor__learning_rate_init": loguniform(1e-4, 1e-2),
                 "mlpregressor__activation": ["relu", "tanh"]}),
    }


# --- metrics (on response time in ms, after undoing the log) -------------------------------------


def metrics(y_ms: np.ndarray, p_ms: np.ndarray) -> dict[str, float]:
    err = p_ms - y_ms
    ape = np.abs(err) / y_ms
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_ms - y_ms.mean()) ** 2))
    return {
        "n": int(len(y_ms)),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "rmse_ms": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(ape)),
        "p90_ape": float(np.quantile(ape, 0.9)),
        "r2_log": float(1 - np.sum((np.log(p_ms) - np.log(y_ms)) ** 2)
                        / np.sum((np.log(y_ms) - np.log(y_ms).mean()) ** 2)),
    }


def bootstrap_ci(y_ms: np.ndarray, p_ms: np.ndarray, n: int = 2000, seed: int = SEED) -> dict[str, list[float]]:
    """95% percentile intervals by resampling test configurations with replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_ms), size=(n, len(y_ms)))
    out = {"r2": [], "rmse_ms": [], "p90_ape": []}
    for i in idx:
        m = metrics(y_ms[i], p_ms[i])
        for k in out:
            out[k].append(m[k])
    return {k: [float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))] for k, v in out.items()}


# --- training --------------------------------------------------------------------------------------


@dataclass
class Fitted:
    name: str
    estimator: object
    best_params: dict
    cv_rmse_log: float  # grouped 5-fold CV, RMSE of log(rt)
    oof_pred_ms: np.ndarray = field(repr=False)  # out-of-fold predictions on the training set

    def predict_ms(self, c: pd.DataFrame) -> np.ndarray:
        return np.exp(self.estimator.predict(features(c)))


def fit_all(train: pd.DataFrame, n_iter: int = 40, folds: int = 5, names: list[str] | None = None) -> dict[str, Fitted]:
    X, y, g = features(train), target(train), groups(train)
    cv = list(GroupKFold(n_splits=folds).split(X, y, g))
    out = {}
    for name, (est, space) in model_specs().items():
        if names and name not in names:
            continue
        search = RandomizedSearchCV(est, space, n_iter=n_iter, cv=cv, scoring="neg_root_mean_squared_error",
                                    random_state=SEED, n_jobs=-1, refit=True)
        search.fit(X, y)
        oof = np.empty(len(y))
        for tr, te in cv:
            m = clone(search.best_estimator_).fit(X.iloc[tr], y[tr])
            oof[te] = m.predict(X.iloc[te])
        out[name] = Fitted(name, search.best_estimator_, search.best_params_, -search.best_score_, np.exp(oof))
    return out


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


def evaluate(fitted: dict[str, Fitted], split: Split, ci: bool = True) -> pd.DataFrame:
    rows = []
    for name, f in fitted.items():
        y_tr = split.train["rt_ms"].to_numpy()
        rows.append({"model": name, "set": "train_oof", "cv_rmse_log": f.cv_rmse_log, **metrics(y_tr, f.oof_pred_ms)})
        for set_name, df in (("val_clean_steady", split.test_clean), ("val_v1_style_steady", split.test_v1_style),
                             ("val_clean_all", split.test_all_clean)):
            y, p = df["rt_ms"].to_numpy(), f.predict_ms(df)
            row = {"model": name, "set": set_name, "cv_rmse_log": f.cv_rmse_log, **metrics(y, p)}
            if ci and set_name == "val_clean_steady":
                b = bootstrap_ci(y, p)
                row |= {"r2_ci": b["r2"], "rmse_ci": b["rmse_ms"], "p90_ape_ci": b["p90_ape"]}
            rows.append(row)
    return pd.DataFrame(rows)
