import pandas as pd
import pytest

from conftest import Cfg, Rep, standard_configs, write_dataset, write_manifest
from csizer import dataset as ds


def test_load_runs_normalises_units(synth):
    r = ds.load_runs(synth)
    assert len(r) == 12
    assert r["memory"].dtype.kind == "i" and set(r["memory"]) == {512}
    assert set(r["device_write"]) == {20}
    assert r.loc[r["size"] == 200, "mb_written"].eq(2.0).all()
    assert r.loc[r["size"] == 50, "mb_written"].eq(0.0).all()
    assert not r["failed"].any()


def test_wrong_columns_rejected(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="unexpected columns"):
        ds.read_runs_csv(p)


def test_config_table_medians_and_ids(synth):
    t = ds.build_config_table(ds.load_runs(synth)).set_index("config_id")
    assert list(t.index) == ["train_0001", "train_0002", "validation_0001", "grid_slice_0001"]
    assert t.loc["train_0001", "rt_ms"] == 5.7
    assert t.loc["train_0002", "rt_spread"] == pytest.approx(25.2 / 24.9)
    assert (t["label"] == "steady").all()
    assert t.loc["grid_slice_0001", "slice_param"] == "arr"


def test_failed_repeat_is_excluded_from_median_and_forces_overload(tmp_path):
    cfgs = [Cfg("train", arr=30, size=200, reps=(Rep(2000.0, thr=20.0), Rep(2200.0, thr=26.0, errors=58, reset=58),
                                                Rep(0.0, thr=0.0, errors=450, refused=450)))]
    t = ds.build_config_table(ds.load_runs(write_dataset(tmp_path / "v1", cfgs))).iloc[0]
    assert t["n_failed"] == 1
    assert t["rt_ms"] == 2100.0  # median of the two answering repeats
    assert t["label"] == "overload"


def test_stall_detection(tmp_path):
    cfgs = [Cfg("train", reps=(Rep(16.0), Rep(16.2), Rep(900.0))),  # one stall
            Cfg("train", reps=(Rep(16.0), Rep(40.0), Rep(45.0))),  # slower but < 50 ms extra: not a stall
            Cfg("train", reps=(Rep(5.7), Rep(300.0), Rep(200.0)))]  # two stalls: the median is a stall
    t = ds.build_config_table(ds.load_runs(write_dataset(tmp_path / "v1", cfgs))).set_index("config_id")
    assert t["n_stalls"].tolist() == [1, 0, 2]
    assert t.loc["train_0001", "rt_ms"] == 16.2
    assert t.loc["train_0003", "rt_ms"] == 200.0


@pytest.mark.parametrize("thr,errors,want", [
    (10.0, 0, "steady"),
    (9.5, 0, "steady"),        # exactly 95% of offered
    (9.4, 0, "near_saturation"),
    (10.0, 1, "near_saturation"),  # any error
    (8.0, 0, "near_saturation"),   # exactly 80%
    (7.9, 0, "overload"),
    (10.0, 30, "overload"),        # 30 / 150 = 20% errors
])
def test_labels(tmp_path, thr, errors, want):
    cfgs = [Cfg("train", arr=10, reps=tuple(Rep(16.0, thr=thr, errors=errors) for _ in range(3)))]
    t = ds.build_config_table(ds.load_runs(write_dataset(tmp_path / "v1", cfgs)))
    assert t.iloc[0]["label"] == want


def test_duplicates_and_validation_leakage(tmp_path):
    cfgs = standard_configs() + [Cfg("train", arr=1, cpuLoad=100_000),  # same tuple as train_0001
                                 Cfg("validation", arr=10, size=200)]  # same tuple as train_0002
    t = ds.build_config_table(ds.load_runs(write_dataset(tmp_path / "v1", cfgs)))
    dups = ds.parameter_duplicates(t)
    assert len(dups) == 2
    assert ds.validation_in_train(t) == ["validation_0002"]


def test_manifest_ok(synth):
    assert ds.verify_manifest(synth) == []


def test_manifest_detects_edit_missing_and_extra(synth):
    f = synth.root / "runs" / "raw_results.csv"
    f.write_text(f.read_text().replace("5.6", "5.5", 1))
    (synth.root / "raw_logs" / "train_0001_rep1.txt").unlink()
    (synth.root / "configs" / "extra.csv").write_text("x\n")
    problems = ds.verify_manifest(synth)
    assert "hash mismatch: runs/raw_results.csv" in problems
    assert "missing: raw_logs/train_0001_rep1.txt" in problems
    assert "not in manifest: configs/extra.csv" in problems


def test_load_logs_matches_runs(synth):
    runs = ds.load_runs(synth)
    logs = ds.load_logs(synth, runs)
    m = runs.merge(logs, on=["config_id", "repeat"])
    assert (m["reply_time_mean_ms"] == m["log_response_ms"]).all()
    assert isinstance(logs, pd.DataFrame) and len(logs) == len(runs)


def test_manifest_helper_roundtrip(synth):
    write_manifest(synth.root)
    assert ds.verify_manifest(synth) == []
