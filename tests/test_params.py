import pytest

from csizer import params


@pytest.mark.parametrize("v,want", [("512m", 512), ("1024m", 1024), ("1024M", 1024), (512, 512)])
def test_parse_memory(v, want):
    assert params.parse_memory(v) == want


@pytest.mark.parametrize("v,want", [("10mb", 10), ("50MB", 50), (20, 20)])
def test_parse_mbps(v, want):
    assert params.parse_mbps(v) == want


@pytest.mark.parametrize("bad", ["512", "1g", "", "mb", "10m"])
def test_bad_units_are_rejected(bad):
    with pytest.raises(ValueError):
        if bad.endswith("m") and not bad.endswith("mb"):
            params.parse_mbps(bad)
        else:
            params.parse_memory(bad)


@pytest.mark.parametrize("size,writes", [(10, 0), (25, 0), (50, 0), (99, 0), (100, 1), (199, 1), (200, 2)])
def test_writes_follow_the_php_loop(size, writes):
    # for ($i = 1; $i <= $size/100; $i++): no iteration when size/100 < 1
    assert params.writes_per_request(size) == writes
    assert params.mb_written_per_request(size) == writes * params.MB_PER_WRITE


def test_nominal_write_differs_from_real_write_below_100():
    assert params.nominal_mb_per_request(50) == 0.5
    assert params.mb_written_per_request(50) == 0.0


def test_negative_size_rejected():
    with pytest.raises(ValueError):
        params.writes_per_request(-1)


def test_levels_cover_every_param_once():
    assert set(params.LEVELS) == set(params.PARAMS)
    assert set(params.WORKLOAD_PARAMS) | set(params.RESOURCE_PARAMS) == set(params.PARAMS)
    full_factorial = 1
    for p in params.PARAMS:
        full_factorial *= len(params.LEVELS[p])
    assert full_factorial == 2880  # 8 x 3 x 2 x 3 x 5 x 4
