#!/usr/bin/env python3
"""
parse_scaling.py  –  Parse weak-scaling log files and plot avg TFLOPs/s vs Forward TFLOPs.

Usage:
    python parse_scaling.py --logs_dir /path/to/logs/
    python parse_scaling.py --logs_dir /path/to/logs/ --out_path results/plot.png --sweep_id my_sweep
"""

import argparse
import glob
import re
import sys
from pathlib import Path
import pandas as pd

import matplotlib

import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D

from utils import KIT_PALETTE


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_log(path: Path) -> dict | None:
    """
    Extract from each log:
      x-axis  forward_tflops : "Forward flops per iteration is X TFLOP"
      y-axis  avg_tflops     : mean of per-epoch "average TFLOPs = X"
      label   embed_dim      : "--embed_dim N"  (optional)
              params         : "Parameters: N"  (optional)
    """
    text = path.read_text()

    # if not re.search("FINITE!", text):
    #     print(f"  [skip] Run did not finish successfully: {path.name}")
    #     return None

    # Slurm configuration
    partition = re.search("SLURM_PARTITION:\s+([a-z\-]+)", text).group(1)
    job_id = int(re.search("SLURM_JOB_ID:\s+(\d+)", text).group(1))
    nnodes = int(re.search("SLURM_NNODES:\s+(\d+)", text).group(1))
    ntasks = int(re.search("SLURM_NTASKS:\s+(\d+)", text).group(1))

    # ── Forward TFLOP per iteration (x-axis) ──────────────────────────────────
    # "Forward flops per iteration is 4.282531261444092 TFLOP"
    m_fwd_bwd = re.search(
        r"forward-backward flops per iteration is\s*([\d.eE+\-]+)\s*TFLOP", text, 
    )
    if not m_fwd_bwd:
        print(f"  [skip] no 'forward-backward flops per iteration' line in {path.name}")
        return None
    forward_backward = float(m_fwd_bwd.group(1))

    # ── Per-epoch average TFLOPs (y-axis source) ──────────────────────────────
    # "Epoch 2 time [min]: 0.547..., average loss 2.25e+17, average TFLOPs = 39.109..."
    timing_pattern = re.compile(
        r"Average time per batch\s*([\d.eE+\-]+)\s*ms",
    )
    batch_timings = np.array([float(tf) for tf in timing_pattern.findall(text) ])

    if len(batch_timings) < 3:
        print(f"  [skip] Ran for less than 50 gradient updates {path.name}")
        return None

    # ── Optional labels ───────────────────────────────────────────────────────
    m_params = re.search(r"Parameters:\s*(\d+)", text)
    params = int(m_params.group(1)) if m_params else None

    # Parse mesh dims
    mesh_str = re.search(r"mesh_dims=[\d-]+,[\d-]+,[\d-]+,[\d-]+,[\d-]+", text)

    if not mesh_str:
        print(f" [skip] Mesh dims not available for {path.name}")

    mesh_dims = [int(dim) for dim in mesh_str.group(0).split("=")[1].split(",")]
    for i, dim in enumerate(mesh_dims):
        if dim == -1:
            tmp = np.abs(np.prod(mesh_dims))
            mesh_dims[i] = ntasks // tmp


    return {
        "file": path.name,
        "partition": partition,
        "job_id": job_id,
        "nodes": nnodes,
        "tasks": ntasks,
        "forward_backward_tflops": forward_backward,
        "avg_batch_timing": batch_timings[1:].mean(),
        "std_batch_timing": batch_timings[1:].std(),
        "params": params,
        "exp_dim": mesh_dims[0],
        "dp_dim": mesh_dims[1],
        "up_dim": mesh_dims[2],
        "jchannel_dim": mesh_dims[3],
        "jspatial_dim": mesh_dims[4],
    }



# ── Plotting ──────────────────────────────────────────────────────────────────


def _fmt_tflop(v, _=None):
    if abs(v) >= 1000:
        return f"{v / 1000:.2g}P"
    return f"{v:.3g}T"


def plot(
    records: list[dict],
    out_path: Path = Path("scaling_plot.png"),
    sweep_id: str | None = None,
    x_axis: str = "Model size",
):

    records = sorted(records, key=lambda r: r["forward_backward_tflops"])
    n = len(records)

    if x_axis == "embedding_dimension":
        xs = np.array([r["forward_backward_tflops"] for r in records])
    else:
        xs = np.array([r[f"{x_axis}"] for r in records])

    ys = np.array([r["avg_tflops"] for r in records])
    labels = [
        (
            f"{x_axis}={r[f'{x_axis}']}"
            if r[f"{x_axis}"] is not None
            else Path(r["file"]).stem
        )
        for r in records
    ]
    # print(f'records {records}')

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.grid(color=KIT_GRAY60, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Trend line
    ax.plot(
        xs,
        ys,
        color=KIT_BLUE,
        linewidth=2.0,
        zorder=2,
        alpha=0.85,
        solid_capstyle="round",
        solid_joinstyle="round",
    )

    # Points + point labels
    for i, (x, y, lbl) in enumerate(zip(xs, ys, labels)):
        color = KIT_PALETTE[i % len(KIT_PALETTE)]
        ax.scatter(
            x, y, color=color, s=90, zorder=4, edgecolors="white", linewidths=0.8
        )
        ax.annotate(
            lbl,
            (x, y),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=13,
            color="black",
            fontfamily="monospace",
        )

    # Spines: hide top/right, style bottom/left
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(0.9)

    # ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_tflop))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.3g}"))
    ax.tick_params(colors="black", labelsize=14, length=4, width=0.8)
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")

    if x_axis == "embedding_dimension":
        ax.set_xlabel("Model size [TFLOP in forward pass]", fontsize=15, labelpad=8)
    else:
        ax.set_xlabel(f"{x_axis}", fontsize=15, labelpad=8)
    ax.set_ylabel("Throughput [TFLOPs/s/node]", fontsize=15, labelpad=8)

    ax.set_title(
        f"{x_axis} vs. Throughput",
        fontsize=20,
        fontweight="bold",
        color=(0.1, 0.1, 0.1),
        pad=14,
        loc="left",
    )

    subtitle_parts = [""]
    if sweep_id:
        subtitle_parts.append(f"sweep: {sweep_id}")
    ax.text(
        0.0,
        0.98,
        "  ·  ".join(subtitle_parts),
        transform=ax.transAxes,
        fontsize=9,
        color="black",
        va="bottom",
    )

    if n <= 10:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=KIT_PALETTE[i % len(KIT_PALETTE)],
                markersize=7,
                label=lbl,
                markeredgecolor="white",
                markeredgewidth=0.6,
            )
            for i, lbl in enumerate(labels)
        ]
        ax.legend(
            handles=handles,
            fontsize=15,
            frameon=True,
            framealpha=0.9,
            edgecolor=KIT_GRAY10,
            loc="best",
            title="runs",
            title_fontsize=15,
        )

    plt.tight_layout(pad=1.4)
    fig.savefig(out_path, dpi=180, facecolor="white", bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def collect_paths(path: str) -> list[Path]:
    p = Path(path)
    if p.is_dir():
        return sorted(p.glob("run_*.out"))
    return sorted(Path(f) for f in glob.glob(path))


def main(
    logs_dir: str,
    out_path: str,
):
    paths = collect_paths(logs_dir)
    if not paths:
        print("No .out files found.")
        sys.exit(1)

    print(f"Parsing {len(paths)} log file(s)...\n")
    records = []
    for p in paths:
        run = parse_log(p)
        if run:
            records.append(run)

    if not records:
        print("No valid records parsed.")
        sys.exit(1)

    df = pd.DataFrame.from_records(records)
    df.info()

    if not out_path:
        p = Path(logs_dir).parent / "results.csv"
    
    df.to_csv(p)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse weak-scaling .out logs and produce a throughput plot."
    )
    parser.add_argument(
        "--logs_dir",
        required=True,
        help="Directory of *.out log files, or a glob pattern.",
    )
    parser.add_argument("--out_path", default=None, help="By default, will choose the parent directory of the logs.")
    args = parser.parse_args()
    main(args.logs_dir, args.out_path)
