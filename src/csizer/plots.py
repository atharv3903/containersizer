"""Figures describing the dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .checks import Bundle  # noqa: E402
from .params import LEVELS, PARAMS  # noqa: E402

COLORS = {"steady": "#2a7ab9", "near_saturation": "#e0a030", "overload": "#c8403a"}


def coverage(b: Bundle, out: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, p in zip(axes.flat, PARAMS, strict=True):
        lv = LEVELS[p]
        x = np.arange(len(lv))
        for i, kind in enumerate(("train", "validation")):
            c = b.configs[b.configs.experiment_type == kind][p].value_counts()
            ax.bar(x + (i - 0.5) * 0.4, [c.get(v, 0) for v in lv], width=0.4, label=kind)
        ax.set_xticks(x, [str(v) for v in lv], rotation=30 if p == "cpuLoad" else 0)
        ax.set_title(p)
        ax.set_ylabel("configurations")
    axes.flat[0].legend()
    fig.suptitle("Configurations per design level (train = LHS, validation = random)")
    fig.tight_layout()
    path = out / "coverage.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def grid_slices(b: Bundle, out: Path) -> Path:
    g = b.configs[b.configs.experiment_type == "grid_slice"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, p in zip(axes.flat, PARAMS, strict=True):
        s = g[g.slice_param == p].sort_values(p)
        ax.plot(range(len(s)), s["rt_ms"], "o-")
        ax.set_xticks(range(len(s)), [str(v) for v in s[p]], rotation=30 if p == "cpuLoad" else 0)
        ax.set_title(f"sweep {p}")
        ax.set_ylabel("mean response time (ms)")
        if s["rt_ms"].max() > 20 * s["rt_ms"].min():
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0, top=max(30.0, s["rt_ms"].max() * 1.15))  # flat effects must look flat
    fig.suptitle("One-at-a-time sweeps around the midpoint (arr 10, cpus 1, 512 MiB, 20 MB/s, size 50, cpuLoad 500k)")
    fig.tight_layout()
    path = out / "grid_slices.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def rt_distribution(b: Bundle, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.logspace(np.log10(b.configs.rt_ms.min()), np.log10(b.configs.rt_ms.max()), 40)
    for lab, c in COLORS.items():
        ax.hist(b.configs[b.configs.label == lab]["rt_ms"], bins=bins, color=c, label=lab, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("mean response time per configuration (ms, log scale)")
    ax.set_ylabel("configurations")
    ax.legend()
    ax.set_title("Response times span four orders of magnitude; overload is the long tail")
    fig.tight_layout()
    path = out / "rt_distribution.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def repeat_spread(b: Bundle, out: Path) -> Path:
    r = b.runs[~b.runs.failed]
    fastest = r.groupby("config_id")["reply_time_mean_ms"].transform("min")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for w, c in zip((0, 1, 2), ("#2a7ab9", "#e0a030", "#c8403a"), strict=True):
        m = r["writes_per_request"] == w
        ax.scatter(fastest[m], r.loc[m, "reply_time_mean_ms"], s=10, color=c, label=f"{w} MB written per request")
    lim = [r["reply_time_mean_ms"].min() * 0.8, r["reply_time_mean_ms"].max() * 1.2]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("fastest repeat of the configuration (ms)")
    ax.set_ylabel("each repeat (ms)")
    ax.legend()
    ax.set_title("Repeats sit on the diagonal; stalls are the points far above it")
    fig.tight_layout()
    path = out / "repeat_spread.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def cpu_cost(b: Bundle, out: Path) -> Path:
    c = b.configs[(b.configs.label == "steady") & (b.configs.writes_per_request == 0)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cpus, mk in zip((0.5, 1.0, 2.0), ("v", "o", "^"), strict=True):
        s = c[c.cpus == cpus]
        ax.scatter(s["cpuLoad"], s["rt_ms"], marker=mk, label=f"cpus {cpus}", alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cpuLoad (rand() calls per request)")
    ax.set_ylabel("mean response time (ms)")
    ax.legend()
    ax.set_title("Steady, no-write configurations: response time follows CPU work per request")
    fig.tight_layout()
    path = out / "cpu_cost.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def all_plots(b: Bundle, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    return [f(b, out) for f in (coverage, grid_slices, rt_distribution, repeat_spread, cpu_cost)]
