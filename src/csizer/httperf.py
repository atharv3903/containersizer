"""Parser for httperf's text report.

httperf prints one summary per run. We read every field we may need later, not only the
four the original collection script kept, so that the raw logs remain the source of truth.
Field meanings (httperf 0.9):

  Total: connections C requests Q replies R test-duration T s
  Connection rate: X conn/s (... <=N concurrent connections)   N = peak simultaneous connections
  Connection time [ms]: min avg max median stddev               connect + request + reply, per connection
  Connection time [ms]: connect X                               TCP handshake only
  Reply rate [replies/s]: min avg max stddev (S samples)        sampled every 5 s; S=0 for runs < 5 s
  Reply time [ms]: response X transfer Y                        X = request sent -> first reply byte
  Reply status: 1xx= 2xx= 3xx= 4xx= 5xx=
  Errors: total E client-timo socket-timo connrefused connreset
  Errors: fd-unavail addrunavail ftab-full other
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

_NUM = r"([-+]?\d+(?:\.\d+)?)"


@dataclass(frozen=True)
class HttperfRun:
    rate: float  # --rate (requests/s offered)
    num_conns: int  # --num-conns
    uri: str
    connections: int
    requests: int
    replies: int
    duration_s: float
    max_concurrent: int
    conn_time_min_ms: float
    conn_time_avg_ms: float
    conn_time_max_ms: float
    conn_time_median_ms: float
    conn_time_stddev_ms: float
    connect_ms: float
    reply_rate_min: float
    reply_rate_avg: float
    reply_rate_max: float
    reply_rate_stddev: float
    reply_rate_samples: int
    response_ms: float  # mean time from request sent to first byte of the reply
    transfer_ms: float
    status_2xx: int
    status_non2xx: int
    errors_total: int
    err_client_timeout: int
    err_socket_timeout: int
    err_conn_refused: int
    err_conn_reset: int
    err_other: int  # fd-unavail + addrunavail + ftab-full + other

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def completed_fraction(self) -> float:
        return self.replies / self.num_conns if self.num_conns else 0.0


class HttperfParseError(ValueError):
    pass


def _search(pattern: str, text: str, name: str) -> re.Match:
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        raise HttperfParseError(f"missing field: {name}")
    return m


def parse_text(text: str) -> HttperfRun:
    cmd = _search(r"^httperf .*$", text, "command line").group(0)
    rate = float(_search(r"--rate=" + _NUM, cmd, "--rate").group(1))
    num_conns = int(_search(r"--num-conns=(\d+)", cmd, "--num-conns").group(1))
    uri = _search(r"--uri=(\S+)", cmd, "--uri").group(1)

    tot = _search(
        rf"^Total: connections (\d+) requests (\d+) replies (\d+) test-duration {_NUM} s", text, "Total"
    )
    conc = _search(r"^Connection rate: .*<=(\d+) concurrent connections", text, "Connection rate")
    ct = _search(
        rf"^Connection time \[ms\]: min {_NUM} avg {_NUM} max {_NUM} median {_NUM} stddev {_NUM}",
        text,
        "Connection time",
    )
    cc = _search(rf"^Connection time \[ms\]: connect {_NUM}", text, "Connection time connect")
    rr = _search(
        rf"^Reply rate \[replies/s\]: min {_NUM} avg {_NUM} max {_NUM} stddev {_NUM} \((\d+) samples\)",
        text,
        "Reply rate",
    )
    rt = _search(rf"^Reply time \[ms\]: response {_NUM} transfer {_NUM}", text, "Reply time")
    st = _search(r"^Reply status: 1xx=(\d+) 2xx=(\d+) 3xx=(\d+) 4xx=(\d+) 5xx=(\d+)", text, "Reply status")
    e1 = _search(
        r"^Errors: total (\d+) client-timo (\d+) socket-timo (\d+) connrefused (\d+) connreset (\d+)",
        text,
        "Errors line 1",
    )
    e2 = _search(
        r"^Errors: fd-unavail (\d+) addrunavail (\d+) ftab-full (\d+) other (\d+)", text, "Errors line 2"
    )
    s = [int(x) for x in st.groups()]
    return HttperfRun(
        rate=rate,
        num_conns=num_conns,
        uri=uri,
        connections=int(tot.group(1)),
        requests=int(tot.group(2)),
        replies=int(tot.group(3)),
        duration_s=float(tot.group(4)),
        max_concurrent=int(conc.group(1)),
        conn_time_min_ms=float(ct.group(1)),
        conn_time_avg_ms=float(ct.group(2)),
        conn_time_max_ms=float(ct.group(3)),
        conn_time_median_ms=float(ct.group(4)),
        conn_time_stddev_ms=float(ct.group(5)),
        connect_ms=float(cc.group(1)),
        reply_rate_min=float(rr.group(1)),
        reply_rate_avg=float(rr.group(2)),
        reply_rate_max=float(rr.group(3)),
        reply_rate_stddev=float(rr.group(4)),
        reply_rate_samples=int(rr.group(5)),
        response_ms=float(rt.group(1)),
        transfer_ms=float(rt.group(2)),
        status_2xx=s[1],
        status_non2xx=s[0] + s[2] + s[3] + s[4],
        errors_total=int(e1.group(1)),
        err_client_timeout=int(e1.group(2)),
        err_socket_timeout=int(e1.group(3)),
        err_conn_refused=int(e1.group(4)),
        err_conn_reset=int(e1.group(5)),
        err_other=sum(int(x) for x in e2.groups()),
    )


def parse_file(path: str | Path) -> HttperfRun:
    try:
        return parse_text(Path(path).read_text())
    except HttperfParseError as e:
        raise HttperfParseError(f"{path}: {e}") from e


FIELD_NAMES = tuple(f.name for f in fields(HttperfRun))
