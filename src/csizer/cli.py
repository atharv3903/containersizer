"""Command line: `csizer verify | report | plots`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import checks, plots, report
from .dataset import DataPaths, default_data_root


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="csizer")
    ap.add_argument("--data", type=Path, default=None, help="dataset root (default: data/v1 or $CSIZER_DATA)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify", help="run every data check; exit 1 if any check fails")
    rp = sub.add_parser("report", help="write the dataset report and derived table")
    rp.add_argument("--out", type=Path, required=True)
    pp = sub.add_parser("plots", help="write dataset figures")
    pp.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    paths = DataPaths(a.data or default_data_root())
    bundle = checks.load_bundle(paths)

    if a.cmd == "plots":
        for p in plots.all_plots(bundle, a.out):
            print(p)
        return 0

    results = checks.run_all(bundle)
    for c in results:
        print(f"[{c.status.upper():4}] {c.name}: {c.summary}")
    if a.cmd == "report":
        report.write(bundle, results, a.out)
        print(f"report written to {a.out}")
    failed = [c for c in results if c.status == "fail"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
