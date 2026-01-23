"""Facts about the real v1 dataset that later stages rely on. Skipped if the data is absent."""

import pytest

from csizer import checks

pytestmark = pytest.mark.realdata


@pytest.fixture(scope="module")
def bundle(real_paths):
    return checks.load_bundle(real_paths)


@pytest.fixture(scope="module")
def results(bundle):
    return {c.name: c for c in checks.run_all(bundle)}


def test_sizes(bundle):
    assert len(bundle.runs) == 975
    assert len(bundle.configs) == 325
    assert bundle.configs["experiment_type"].value_counts().to_dict() == {"train": 250, "validation": 50,
                                                                           "grid_slice": 25}


def test_integrity_checks_all_pass(results):
    for name in ("manifest", "parts_union", "configs_match", "repeats", "on_grid", "logs_agree"):
        assert results[name].status == "ok", results[name].summary
    assert not [c for c in results.values() if c.status == "fail"]


def test_one_dead_repeat(bundle):
    dead = bundle.runs[bundle.runs.failed]
    assert dead[["config_id", "repeat"]].values.tolist() == [["validation_0019", 3]]


def test_labels(bundle):
    assert bundle.configs["label"].value_counts().to_dict() == {"steady": 316, "overload": 5, "near_saturation": 4}


def test_six_validation_configs_were_also_training_configs(results):
    assert len(results["duplicates"].details["validation_in_train"]) == 6


def test_every_stall_is_in_a_config_that_writes_to_disk(bundle):
    stalled = bundle.configs[bundle.configs.n_stalls > 0]
    assert len(stalled) == 13
    assert (stalled["writes_per_request"] > 0).all()


def test_sizes_below_100_behave_identically(results):
    rt = results["zero_write_sizes"].details["grid_rt_by_size"]
    assert max(rt[10], rt[25], rt[50]) - min(rt[10], rt[25], rt[50]) < 0.5
    assert rt[200] > rt[50] + 5


def test_cpu_cost(results):
    assert 20 < results["cpu_cost_per_rand"].details["ns_per_rand"] < 30


def test_design_overestimated_saturation(results):
    d = results["design_vs_outcome"].details
    assert (d["planned_near_saturation"], d["measured_non_steady"]) == (60, 8)
    # every non-steady train/validation config is one where the disk term dominates
    assert all(rho_c >= 0.6 for _, _, _, rho_c in d["non_steady_with_rho"])
