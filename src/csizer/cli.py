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
    mp = sub.add_parser("models", help="train and evaluate the five regressors (stage 2)")
    mp.add_argument("--out", type=Path, required=True)
    mp.add_argument("--n-iter", type=int, default=40)
    sp = sub.add_parser("stability", help="utilisation gate vs classifiers (stage 3)")
    sp.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    if a.cmd == "stability":
        from . import stage3

        r = stage3.run(a.data or default_data_root(), a.out)
        print(f"gate held-out recall {r['holdout']['gate']['recall']:.0%}, "
              f"false alarms {r['holdout']['gate']['fp']}; report at {a.out / 'stability_report.md'}")
        return 0

    if a.cmd == "models":
        from . import stage2

        root = a.data or default_data_root()
        (a.out / "plots").mkdir(parents=True, exist_ok=True)
        r = stage2.run(root, a.out, n_iter=a.n_iter)
        stage2.feature_importance(root, a.out)
        print(f"best model: {r['best']}; report at {a.out / 'models_report.md'}")
        return 0

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
