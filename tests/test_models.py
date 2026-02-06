import numpy as np
import pandas as pd
import pytest

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
