import pytest

from csizer import httperf

OK_LOG = """httperf: warning: open file limit > FD_SETSIZE; limiting max. # of open files to FD_SETSIZE
httperf --timeout=120 --client=0/1 --server=192.0.2.1 --port=80 --uri=/server.php?size=10&cpuLoad=100000 --rate=1 --send-buffer=4096 --recv-buffer=16384 --num-conns=50 --num-calls=1
Maximum connect burst length: 1

Total: connections 50 requests 50 replies 50 test-duration 49.007 s

Connection rate: 1.0 conn/s (980.1 ms/conn, <=1 concurrent connections)
Connection time [ms]: min 5.4 avg 6.2 max 7.3 median 6.5 stddev 0.3
Connection time [ms]: connect 0.6
Connection length [replies/conn]: 1.000

Request rate: 1.0 req/s (980.1 ms/req)
Request size [B]: 100.0

Reply rate [replies/s]: min 1.0 avg 1.0 max 1.0 stddev 0.0 (9 samples)
Reply time [ms]: response 5.6 transfer 0.0
Reply size [B]: header 174.0 content 67.0 footer 0.0 (total 241.0)
Reply status: 1xx=0 2xx=50 3xx=0 4xx=0 5xx=0

CPU time [s]: user 9.44 system 39.56 (user 19.3% system 80.7% total 100.0%)
Net I/O: 0.3 KB/s (0.0*10^6 bps)

Errors: total 0 client-timo 0 socket-timo 0 connrefused 0 connreset 0
Errors: fd-unavail 0 addrunavail 0 ftab-full 0 other 0
"""

DEAD_LOG = """httperf --timeout=120 --client=0/1 --server=192.0.2.1 --port=80 --uri=/server.php?size=200&cpuLoad=100000 --rate=30 --send-buffer=4096 --recv-buffer=16384 --num-conns=450 --num-calls=1
Total: connections 450 requests 0 replies 0 test-duration 14.967 s
Connection rate: 30.1 conn/s (33.3 ms/conn, <=1 concurrent connections)
Connection time [ms]: min 0.0 avg 0.0 max 0.0 median 0.0 stddev 0.0
Connection time [ms]: connect 0.5
Reply rate [replies/s]: min 0.0 avg 0.0 max 0.0 stddev 0.0 (2 samples)
Reply time [ms]: response 0.0 transfer 0.0
Reply status: 1xx=0 2xx=0 3xx=0 4xx=0 5xx=0
Errors: total 450 client-timo 0 socket-timo 0 connrefused 450 connreset 0
Errors: fd-unavail 0 addrunavail 0 ftab-full 0 other 0
"""


def test_parses_a_normal_run():
    r = httperf.parse_text(OK_LOG)
    assert (r.rate, r.num_conns, r.replies, r.max_concurrent) == (1.0, 50, 50, 1)
    assert r.uri == "/server.php?size=10&cpuLoad=100000"
    assert r.response_ms == 5.6
    assert (r.conn_time_avg_ms, r.conn_time_max_ms, r.conn_time_stddev_ms) == (6.2, 7.3, 0.3)
    assert (r.reply_rate_avg, r.reply_rate_samples) == (1.0, 9)
    assert r.duration_s == pytest.approx(49.007)
    assert r.status_2xx == 50 and r.status_non2xx == 0
    assert r.errors_total == 0
    assert r.completed_fraction == 1.0


def test_parses_a_run_where_the_server_was_down():
    r = httperf.parse_text(DEAD_LOG)
    assert r.replies == 0 and r.response_ms == 0.0
    assert r.errors_total == 450 and r.err_conn_refused == 450
    assert r.completed_fraction == 0.0


def test_missing_field_is_an_error_naming_the_field():
    broken = OK_LOG.replace("Reply time [ms]: response 5.6 transfer 0.0\n", "")
    with pytest.raises(httperf.HttperfParseError, match="Reply time"):
        httperf.parse_text(broken)


def test_parse_file_reports_the_path(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("garbage")
    with pytest.raises(httperf.HttperfParseError, match="x.txt"):
        httperf.parse_file(p)


def test_field_names_cover_the_dataclass():
    assert "response_ms" in httperf.FIELD_NAMES and len(httperf.FIELD_NAMES) == 29
