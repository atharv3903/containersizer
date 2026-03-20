import pandas as pd
import pytest

from csizer import stability as st


def _c(**kw):
    base = dict(config_id="train_0001", experiment_type="train", arr=10, cpus=1.0, memory=512, device_write=20,
                size=50, cpuLoad=1_000_000, mb_written=0.0, writes_per_request=0, rt_ms=28.0, label="steady",
                n_stalls=0)
    return base | kw


def test_calibration_recovers_a_linear_cpu_law():
    rows = [_c(config_id=f"train_{i:04d}", cpuLoad=cl, rt_ms=3.0 + cl * 25e-6)
            for i, cl in enumerate([100_000, 500_000, 1_000_000] * 4)]
    cal = st.calibrate_cpu(pd.DataFrame(rows))
    assert cal.intercept_ms == pytest.approx(3.0)
    assert cal.ns_per_rand == pytest.approx(25.0)


def test_utilisation_law():
    cal = st.CpuCalibration(intercept_ms=0.0, ns_per_rand=25.0)
    c = pd.DataFrame([_c(arr=20, cpus=0.5, cpuLoad=1_000_000, mb_written=2.0, device_write=50)])
    u = st.utilisation(c, cal)
    assert u.loc[0, "rho_cpu"] == pytest.approx(20 * 0.025 / 0.5)  # 1.0
    assert u.loc[0, "rho_io"] == pytest.approx(20 * 2 / 50)  # 0.8


def test_cpu_bound_failure_does_not_set_the_disk_threshold_to_zero():
    rows = [_c(config_id=f"train_{i:04d}", cpuLoad=cl, rt_ms=3.0 + cl * 25e-6)
            for i, cl in enumerate([100_000, 500_000, 1_000_000] * 3)]
    rows.append(_c(config_id="train_0100", cpuLoad=5_000_000, arr=10, label="overload", rt_ms=20000.0))
    gate = st.fit_gate(pd.DataFrame(rows))
    assert gate.theta_io == 1.0  # theory default, not 0
    rows.append(_c(config_id="train_0101", arr=10, mb_written=1.0, writes_per_request=1, device_write=20,
                   cpuLoad=100_000, label="near_saturation", rt_ms=500.0))
    assert st.fit_gate(pd.DataFrame(rows)).theta_io == pytest.approx(0.5)


def test_gate_flags_either_resource():
    gate = st.Gate(st.CpuCalibration(0.0, 25.0), theta_cpu=1.0, theta_io=0.6)
    c = pd.DataFrame([
        _c(arr=10, cpuLoad=100_000),                                    # idle: safe
        _c(arr=50, cpus=1.0, cpuLoad=1_000_000),                         # rho_cpu 1.25: unsafe
        _c(arr=10, mb_written=2.0, device_write=20, cpuLoad=100_000),     # rho_io 1.0: unsafe
    ])
    assert gate.predict_unsafe(c).tolist() == [False, True, True]
