"""Writes the dataset report (Markdown + JSON) and the derived per-configuration table."""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .checks import Bundle, Check
from .params import LEVELS, PARAMS

ICON = {"ok": "OK", "warn": "WARN", "fail": "FAIL", "info": "INFO"}


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def coverage_tables(b: Bundle) -> dict[str, pd.DataFrame]:
    out = {}
    for kind in ("train", "validation"):
        c = b.configs[b.configs.experiment_type == kind]
        rows = []
        for p in PARAMS:
            counts = c[p].value_counts()
            rows.append({"param": p, **{str(lv): int(counts.get(lv, 0)) for lv in LEVELS[p]}})
        out[kind] = pd.DataFrame(rows).set_index("param")
    return out


def write(b: Bundle, checks: list[Check], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    derived = out_dir / "configs_table.csv"
    b.configs.round(4).to_csv(derived, index=False)

    meta = {
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "csizer_version": __version__,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "data_root": str(b.paths.root),
        "runs": len(b.runs),
        "configurations": len(b.configs),
    }
    payload = {"meta": meta, "checks": [c.__dict__ for c in checks]}
    (out_dir / "dataset_report.json").write_text(json.dumps(payload, indent=2, default=_jsonable))

    lines = [
        "# Dataset report (generated)",
        "",
        f"Generated {meta['generated_utc']} by `csizer report` {meta['csizer_version']} "
        f"(Python {meta['python']}, pandas {meta['pandas']}). Do not edit by hand.",
        "",
        f"Data: `{meta['data_root']}`: {meta['runs']} runs, {meta['configurations']} configurations.",
        "",
        "## Checks",
        "",
        "| Status | Check | Result |",
        "|---|---|---|",
    ]
    lines += [f"| {ICON[c.status]} | `{c.name}` | {c.summary} |" for c in checks]
    lines += ["", "## Coverage of design levels (configurations per level)", ""]
    for kind, t in coverage_tables(b).items():
        lines += [f"### {kind}", "", "| param | " + " | ".join(f"L{i + 1}" for i in range(8)) + " |",
                  "|---|" + "---|" * 8]
        for p, row in t.iterrows():
            cells = [f"{lv}: {int(row[str(lv)])}" for lv in LEVELS[p]]
            lines.append(f"| {p} | " + " | ".join(cells + [""] * (8 - len(cells))) + " |")
        lines.append("")
    lines += ["## Response time by stability label (configuration medians, ms)", "",
              "| label | n | min | p50 | p90 | max |", "|---|---|---|---|---|---|"]
    for lab, g in b.configs.groupby("label"):
        q = g["rt_ms"].quantile([0, 0.5, 0.9, 1]).values
        lines.append(f"| {lab} | {len(g)} | {q[0]:.1f} | {q[1]:.1f} | {q[2]:.1f} | {q[3]:.1f} |")
    lines += ["", "## Details", ""]
    for c in checks:
        if c.details:
            lines += [f"### `{c.name}`", "", "```json", json.dumps(c.details, indent=1, default=_jsonable)[:4000],
                      "```", ""]
    (out_dir / "dataset_report.md").write_text("\n".join(lines) + "\n")
    return payload
