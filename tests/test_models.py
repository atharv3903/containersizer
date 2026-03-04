import numpy as np
import pandas as pd
import pytest

from csizer import dataset as ds
from csizer import models


def _cfg(**kw):
    base = dict(arr=10, cpus=1.0, memory=512, device_write=20, size=200, cpuLoad=1_000_000, mb_written=2.0,
                rt_ms=30.0, label="steady", experiment_type="train", n_stalls=0, config_id="train_0001")
    return base | kw


def test_features_are_physical_and_finite():
    c = pd.DataFrame([_cfg(), _cfg(size=50, mb_written=0.0, cpus=0.5)])
    f = models.features(c)
    assert list(f.columns) == list(models.FEATURES)
    assert np.isfinite(f.to_numpy()).all()
    assert f.loc[0, "rho_io"] == pytest.approx(10 * 2 / 20)
    assert f.loc[1, "rho_io"] == 0.0
    assert f.loc[1, "log_cpu_work_per_cpu"] == pytest.approx(np.log(2_000_000))


def test_duplicate_tuples_share_a_group():
    c = pd.DataFrame([_cfg(), _cfg(config_id="train_0002"), _cfg(arr=20, config_id="train_0003")])
    g = models.groups(c)
    assert g[0] == g[1] != g[2]


def test_metrics_known_values():
    y = np.array([10.0, 20.0, 30.0])
    m = models.metrics(y, y)
    assert m["r2"] == 1.0 and m["rmse_ms"] == 0.0 and m["p90_ape"] == 0.0
    m = models.metrics(y, np.array([11.0, 18.0, 30.0]))
    assert m["rmse_ms"] == pytest.approx(np.sqrt(5 / 3))
    assert m["mape"] == pytest.approx((0.1 + 0.1 + 0) / 3)
    assert models.metrics(y, np.full(3, 20.0))["r2"] == pytest.approx(0.0)  # predicting the mean


def test_bootstrap_is_deterministic_and_brackets_the_estimate():
    rng = np.random.default_rng(1)
    y = rng.uniform(5, 100, 60)
    p = y * rng.normal(1, 0.05, 60)
    a, b = models.bootstrap_ci(y, p, n=500), models.bootstrap_ci(y, p, n=500)
    assert a == b
    r2 = models.metrics(y, p)["r2"]
    assert a["r2"][0] <= r2 <= a["r2"][1]


def test_split_excludes_leaked_and_non_steady_and_stalled():
    c = pd.DataFrame([
        _cfg(config_id="train_0001"),
        _cfg(config_id="train_0002", label="overload"),
        _cfg(config_id="train_0003", n_stalls=2),
        _cfg(config_id="validation_0001", experiment_type="validation"),
        _cfg(config_id="validation_0002", experiment_type="validation"),
        _cfg(config_id="validation_0003", experiment_type="validation", label="overload"),
    ])
    s = models.make_split(c, leaked=["validation_0002"], drop_stalled_median=True)
    assert s.train.config_id.tolist() == ["train_0001"]
    assert s.test_clean.config_id.tolist() == ["validation_0001"]
    assert s.test_v1_style.config_id.tolist() == ["validation_0001", "validation_0002"]
    assert s.test_all_clean.config_id.tolist() == ["validation_0001", "validation_0003"]
    assert models.make_split(c, leaked=[]).train.config_id.tolist() == ["train_0001", "train_0003"]


def test_fit_learns_a_cpu_law_on_synthetic_data():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(80):
        cl = int(rng.choice([100_000, 500_000, 1_000_000, 5_000_000]))
        cpus = float(rng.choice([1.0, 2.0]))
        rows.append(_cfg(config_id=f"train_{i:04d}", cpuLoad=cl, cpus=cpus, arr=int(rng.choice([1, 2, 5])),
                         rt_ms=3.4 + cl * 24.5e-6 * rng.normal(1, 0.01)))
    c = pd.DataFrame(rows)
    f = models.fit_all(c, n_iter=3, folds=4, names=["ridge", "random_forest"])
    for name in ("ridge", "random_forest"):
        m = models.metrics(c.rt_ms.to_numpy(), f[name].oof_pred_ms)
        assert m["r2_log"] > 0.95, (name, m)


@pytest.mark.realdata
def test_real_random_forest_beats_r2_098_on_clean_steady_validation(real_paths):
    c = ds.build_config_table(ds.load_runs(real_paths))
    s = models.make_split(c, ds.validation_in_train(c), drop_stalled_median=True)
    assert (len(s.train), len(s.test_clean)) == (245, 41)
    f = models.fit_all(s.train, n_iter=8, names=["random_forest"])["random_forest"]
    m = models.metrics(s.test_clean.rt_ms.to_numpy(), f.predict_ms(s.test_clean))
    assert m["r2"] > 0.98 and m["p90_ape"] < 0.10
    # and the same model cannot predict overload: R^2 over every label collapses
    a = models.metrics(s.test_all_clean.rt_ms.to_numpy(), f.predict_ms(s.test_all_clean))
    assert a["r2"] < 0.1
