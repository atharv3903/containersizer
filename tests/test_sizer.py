import numpy as np
import pandas as pd
import pytest

from csizer import dataset as ds
from csizer import sizer as sz
from csizer import stability as st


class FakeRegressor:
    """rt = 3 ms + 25 ns x cpuLoad, slower on half a CPU: enough to exercise the sizer's logic."""

    def predict_ms(self, c):
        base = 3.0 + c["cpuLoad"].to_numpy() * 25e-6
        return np.where(c["cpus"].to_numpy() < 1, base * 1.5, base)


def _sizer(guard=0.0):
    gate = st.Gate(st.CpuCalibration(3.0, 25.0), theta_cpu=1.0, theta_io=0.6)
    return sz.Sizer(gate, FakeRegressor(), guard)


def test_containers_are_the_18_options_cheapest_first():
    c = sz.containers()
    assert len(c) == 18
    assert c["cost"].is_monotonic_increasing
    assert tuple(c.iloc[0][["cpus", "memory", "device_write"]]) == (0.5, 512, 10)


def test_recommend_picks_the_cheapest_option_that_is_safe_and_fast_enough():
    s = _sizer()
    best, table = s.recommend(arr=10, size=50, cpuLoad=1_000_000, sla_ms=30)
    # 0.5 CPU: predicted 42 ms > 30; 1 CPU: 28 ms, rho_cpu = 10 x 28 ms / 1 = 0.28 -> safe
    assert (best.cpus, best.memory, best.device_write) == (1.0, 512, 10)
    assert table["meets"].sum() == 12


def test_gate_overrides_a_fast_prediction():
    s = _sizer()
    # 40 req/s x 28 ms = 1.12 CPU-seconds per second: unsafe on 1 CPU even though one request takes 28 ms
    best, _ = s.recommend(arr=40, size=50, cpuLoad=1_000_000, sla_ms=100)
    assert best.cpus == 2.0
    # heavy writes: 30 req/s x 2 MB = 60 MB/s; only the 50 MB/s cap has rho_io 1.2 -> unsafe everywhere
    best, _ = s.recommend(arr=30, size=200, cpuLoad=100_000, sla_ms=100)
    assert best is None


def test_guard_band_makes_the_sizer_more_conservative():
    best0, _ = _sizer(0.0).recommend(arr=1, size=50, cpuLoad=1_000_000, sla_ms=28.5)
    best1, _ = _sizer(0.05).recommend(arr=1, size=50, cpuLoad=1_000_000, sla_ms=28.5)
    assert best0 is not None and best1 is None  # 28 ms fits 28.5; 28 x 1.05 = 29.4 does not


def test_capacity_is_the_last_rate_before_failure():
    s = _sizer()
    # 1 CPU, 28 ms per request: rho_cpu < 1 up to 35 req/s
    assert s.capacity(1.0, 512, 20, 50, 1_000_000, sla_ms=100) == 35
    assert s.capacity(2.0, 512, 20, 50, 1_000_000, sla_ms=100) == 50  # capped at the measured range
    assert s.capacity(1.0, 512, 20, 50, 1_000_000, sla_ms=10) == 0


def _assessed(rows):
    base = dict(arr=10, mb_written=0.0, cpuLoad=100_000, memory=512, device_write=20, label="steady", safe=True)
    return pd.DataFrame([base | r for r in rows])


def test_decision_table_counts_bad_accepts():
    a = _assessed([
        dict(cpus=0.5, rt_ms=8.0, pred_rt_ms=8.0, bound_rt_ms=8.0),
        dict(cpus=1.0, rt_ms=12.0, pred_rt_ms=9.0, bound_rt_ms=9.5),  # accepted at 10 ms, really 12: bad accept
        dict(cpus=2.0, rt_ms=900.0, label="overload", safe=False, pred_rt_ms=6.0, bound_rt_ms=6.0),
    ])
    d = sz.decision_table(a, slas=(10,)).set_index("method")
    assert d.loc["sizer", "bad_accepts"] == 1 and d.loc["sizer", "accepted"] == 2
    assert d.loc["regressor_only", "bad_accepts"] == 2  # also accepts the overloaded one


def test_workload_choices_judges_picks_by_measurement():
    a = _assessed([
        dict(cpus=0.5, rt_ms=900.0, label="overload", safe=False, pred_rt_ms=8.0, bound_rt_ms=8.0),
        dict(cpus=1.0, rt_ms=9.0, pred_rt_ms=9.0, bound_rt_ms=9.3),
    ])
    w = sz.workload_choices(a, slas=(10,)).set_index("method")
    assert w.loc["sizer", "outcome"] == "meets_sla" and bool(w.loc["sizer", "optimal"])
    assert w.loc["regressor_only", "outcome"] == "violates_sla"  # picked the cheap overloaded container


@pytest.mark.realdata
def test_real_sizer_never_picks_a_container_that_breaks_the_sla(real_paths):
    c = ds.build_config_table(ds.load_runs(real_paths))
    a = sz.oof_assessments(c, n_iter=5)
    w = sz.workload_choices(a)
    s, r = w[w.method == "sizer"], w[w.method == "regressor_only"]
    assert (s.outcome == "violates_sla").sum() == 0
    assert (s.outcome == "meets_sla").mean() > 0.9
    assert (r.outcome == "violates_sla").sum() > 0  # the naive design does break SLAs
