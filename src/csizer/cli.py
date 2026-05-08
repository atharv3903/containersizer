"""Command-line interface for dataset checks, evaluation and sizing."""

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
    zp = sub.add_parser("sizer", help="offline evaluation of the sizer (stage 4)")
    zp.add_argument("--out", type=Path, required=True)
    rp2 = sub.add_parser("recommend", help="cheapest container for a workload and SLA")
    for k in ("arr", "size", "cpuLoad"):
        rp2.add_argument(f"--{k}", type=int, required=True)
    rp2.add_argument("--sla-ms", type=float, required=True)
    cp = sub.add_parser("capacity", help="maximum supported arrival rate for a container and SLA")
    cp.add_argument("--cpus", type=float, required=True)
    cp.add_argument("--memory-mib", type=int, required=True)
    cp.add_argument("--device-write", type=int, required=True, help="disk write limit in MB/s")
    cp.add_argument("--size", type=int, required=True)
    cp.add_argument("--cpuLoad", type=int, required=True)
    cp.add_argument("--sla-ms", type=float, required=True)
    cp.add_argument("--max-arr", type=int, default=50, help="maximum request rate to test")
    a = ap.parse_args(argv)
    if a.cmd == "capacity" and a.max_arr < 1:
        cp.error("--max-arr must be at least 1")

    if a.cmd == "sizer":
        from . import stage4

        r = stage4.run(a.data or default_data_root(), a.out)
        print(f"sizer: {r['workload_summary']['sizer']}; report at {a.out / 'sizer_report.md'}")
        return 0

    if a.cmd in {"recommend", "capacity"}:
        from . import dataset as ds
        from . import sizer as sz

        c = ds.build_config_table(ds.load_runs(DataPaths(a.data or default_data_root())))
        s = sz.fit_sizer(c[c.experiment_type == "train"])
        if a.cmd == "capacity":
            rate = s.capacity(a.cpus, a.memory_mib, a.device_write, a.size, a.cpuLoad, a.sla_ms, a.max_arr)
            print(f"capacity: {rate} req/s")
            return 0
        best, table = s.recommend(a.arr, a.size, a.cpuLoad, a.sla_ms)
        cols = ["cpus", "memory", "device_write", "cost", "margin", "safe", "pred_rt_ms", "bound_rt_ms", "meets"]
        print(table[cols].round(3).to_string(index=False))
        print("recommendation:", "none" if best is None else
              f"{best.cpus:g} CPU, {best.memory} MiB, {best.device_write} MB/s")
        return 0

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
