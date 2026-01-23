"""Every integrity check passes on healthy data and fails on the specific damage it guards against."""

from conftest import Cfg, Rep, standard_configs, write_dataset, write_manifest
from csizer import checks
from csizer.dataset import DataPaths

INTEGRITY = ("manifest", "parts_union", "configs_match", "repeats", "on_grid", "logs_agree")


def statuses(paths: DataPaths) -> dict[str, str]:
    b = checks.load_bundle(paths)
    fns = [checks.check_manifest, checks.check_parts, checks.check_configs_match, checks.check_repeats,
           checks.check_grid, checks.check_logs_agree, checks.check_failed_repeats, checks.check_repeat_agreement,
           checks.check_duplicates, checks.check_labels]
    return {c.name: c.status for c in (f(b) for f in fns)}


def edit(paths: DataPaths, rel: str, old: str, new: str, count: int = 1) -> None:
    f = paths.root / rel
    text = f.read_text()
    assert old in text, (rel, old)
    f.write_text(text.replace(old, new, count))
    write_manifest(paths.root)  # keep the manifest consistent so only the targeted check fires


def test_healthy_dataset_passes_every_check(synth):
    s = statuses(synth)
    assert all(s[k] == "ok" for k in INTEGRITY), s
    assert s["errors"] == "ok" and s["repeat_agreement"] == "ok" and s["duplicates"] == "ok"


def test_manifest_check_fails_on_unrecorded_edit(synth):
    f = synth.root / "raw_logs" / "train_0001_rep1.txt"
    f.write_text(f.read_text() + "\n")
    assert statuses(synth)["manifest"] == "fail"


def test_csv_value_that_disagrees_with_its_log_fails(synth):
    for rel in ("runs/raw_results.csv", "runs/raw_results_train.csv"):
        edit(synth, rel, "train_0001,1,1,1.0,512m,20mb,50,100000,,1.0,5.6", "train_0001,1,1,1.0,512m,20mb,50,100000,,1.0,5.9")
    s = statuses(synth)
    assert s["logs_agree"] == "fail" and s["parts_union"] == "ok"


def test_union_file_that_differs_from_parts_fails(synth):
    f = synth.root / "runs" / "raw_results.csv"
    lines = f.read_text().splitlines()
    f.write_text("\n".join(lines[:-1]) + "\n")
    write_manifest(synth.root)
    assert statuses(synth)["parts_union"] == "fail"


def test_missing_repeat_fails(tmp_path):
    cfgs = standard_configs()
    cfgs[0] = Cfg("train", arr=1, cpuLoad=100_000, reps=(Rep(5.6), Rep(5.7)))
    assert statuses(write_dataset(tmp_path / "v1", cfgs))["repeats"] == "fail"


def test_off_grid_value_fails(tmp_path):
    cfgs = standard_configs()
    cfgs[0] = Cfg("train", arr=3)
    assert statuses(write_dataset(tmp_path / "v1", cfgs))["on_grid"] == "fail"


def test_planned_config_never_run_fails(synth):
    f = synth.root / "configs" / "configs_train.csv"
    f.write_text(f.read_text() + "50,2.0,1024m,50mb,10,100000\n")
    write_manifest(synth.root)
    assert statuses(synth)["configs_match"] == "fail"


def test_errors_and_stalls_and_leakage_warn(tmp_path):
    cfgs = standard_configs() + [
        Cfg("train", arr=30, size=200, reps=(Rep(16.0), Rep(16.1), Rep(0.0, thr=0.0, errors=450, refused=450))),
        Cfg("train", arr=20, size=100, reps=(Rep(28.0), Rep(28.1), Rep(900.0))),
        Cfg("validation", arr=1, cpuLoad=100_000),
    ]
    s = statuses(write_dataset(tmp_path / "v1", cfgs))
    assert s["errors"] == "warn" and s["repeat_agreement"] == "warn" and s["duplicates"] == "warn"
    assert all(s[k] == "ok" for k in INTEGRITY)
