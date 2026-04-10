import numpy as np

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


def test_capacity_is_the_last_rate_before_failure():
    s = _sizer()
    # 1 CPU, 28 ms per request: rho_cpu < 1 up to 35 req/s
    assert s.capacity(1.0, 512, 20, 50, 1_000_000, sla_ms=100) == 35
    assert s.capacity(2.0, 512, 20, 50, 1_000_000, sla_ms=100) == 50  # capped at the measured range
    assert s.capacity(1.0, 512, 20, 50, 1_000_000, sla_ms=10) == 0
