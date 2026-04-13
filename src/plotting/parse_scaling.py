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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import KIT_GRAY60, init_matplotlib, set_size

# ── Parsing ───────────────────────────────────────────────────────────────────


def parse_log(path: Path) -> list[dict[str, any]] | None:
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
        r"forward-backward flops per iteration is\s*([\d.eE+\-]+)\s*TFLOP",
        text,
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
    batch_timings = np.array([float(tf) for tf in timing_pattern.findall(text)])

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

    records = []
    for timing in batch_timings[1:]:
        records.append(
            {
                "file": path.name,
                "partition": partition,
                "job_id": job_id,
                "nodes": nnodes,
                "tasks": ntasks,
                "forward_backward_tflops": forward_backward,
                "batch_timing": timing,
                "params": params,
                "exp_dim": mesh_dims[0],
                "dp_dim": mesh_dims[1],
                "up_dim": mesh_dims[2],
                "jchannel_dim": mesh_dims[3],
                "jspatial_dim": mesh_dims[4],
            }
        )
    return records


def process_records(df: pd.DataFrame) -> pd.DataFrame:
    df.rename(
        columns={
            "nodes": "Nodes",
            "batch_timing": "Mean Batch Time (ms)",
        },
        inplace=True,
    )

    df["TFLOPs/s/Node"] = df["forward_backward_tflops"] / (
        df["Mean Batch Time (ms)"] * 1e-3
    )
    df["TFLOPs/s"] = df["TFLOPs/s/Node"] * df["Nodes"]
    df["Samples/s"] = df["dp_dim"] / (df["Mean Batch Time (ms)"] * 1e-3)

    return df


def plot(
    df: pd.DataFrame,
    out_path: Path = Path("scaling_plot.png"),
    show: bool = False,
):

    init_matplotlib()
    fig_size = set_size(width="thesis", fraction=1.0)

    x_ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    x_ticks = [x for x in x_ticks if df["Nodes"].min() <= x <= df["Nodes"].max()]
    print(f"x_ticks: {x_ticks}")

    # Batch timing plot
    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(
        data=df,
        x="Nodes",
        y="Mean Batch Time (ms)",
        marker="o",
        ax=ax,
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)
    ax.set_ylim(bottom=0, top=df["Mean Batch Time (ms)"].max() * 1.2)

    ax.grid(True, which="major", ls="--", lw=0.5)
    if show:
        plt.show()
    else:
        plt.savefig(out_path / "batch_time.png")

    # TFLOPs/s plot
    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(
        data=df,
        x="Nodes",
        y="TFLOPs/s",
        marker="o",
        ax=ax,
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)

    ax.set_yscale("log", base=10)
    ax.set_ylim(top=df["TFLOPs/s"].max() * 1.2)
    ax.grid(True, which="major", ls="--", lw=0.5)

    if show:
        plt.show()
    else:
        plt.savefig(out_path / "total_tflopss.png")

    # Samples/s plot
    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(
        data=df,
        x="Nodes",
        y="Samples/s",
        marker="o",
        ax=ax,
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)
    # ax.set_yscale("log", base=10)
    ax.set_ylim(bottom=0, top=df["Samples/s"].max() * 1.2)
    ax.grid(True, which="major", ls="--", lw=0.5)

    if show:
        plt.show()
    else:
        plt.savefig(out_path / "samples.png")

    # TFLOPs/s/Node plot
    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(
        data=df,
        x="Nodes",
        y="TFLOPs/s/Node",
        marker="o",
        ax=ax,
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)
    # ax.set_yscale("log", base=10)
    ax.set_ylim(bottom=0, top=df["TFLOPs/s/Node"].max() * 1.2)

    ax.grid(True, which="major", ls="--", lw=0.5)

    if show:
        plt.show()
    else:
        plt.savefig(out_path / "tflops_per_node.png")

    # Weak scaling plot (batch_time(N) / batch time(1)  )
    weak_df = df.groupby("Nodes").agg({"Mean Batch Time (ms)": "mean"}).reset_index()
    weak_df["Weak Scaling Efficiency"] = (
        weak_df["Mean Batch Time (ms)"] / weak_df["Mean Batch Time (ms)"].iloc[0]
    )

    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(
        data=weak_df,
        x="Nodes",
        y="Weak Scaling Efficiency",
        marker="o",
        ax=ax,
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)

    ax.set_ylim(bottom=0, top=weak_df["Weak Scaling Efficiency"].max() * 1.2)
    ax.grid(True, which="major", ls="--", lw=0.5)

    if show:
        plt.show()
    else:
        plt.savefig(out_path / "weak_scaling_efficiency.png")

    # Strong scaling plot (Speedup = samples/s(N) / samples/s(1)  )
    strong_df = df.groupby("Nodes").agg({"Samples/s": "mean"}).reset_index()
    strong_df["Speedup"] = strong_df["Samples/s"] / strong_df["Samples/s"].iloc[0]
    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(
        data=strong_df,
        x="Nodes",
        y="Speedup",
        marker="o",
        ax=ax,
    )

    baseline_nodes = strong_df["Nodes"].iloc[0]

    plt.plot(
        x_ticks,
        x_ticks // baseline_nodes,
        ls="--",
        color=KIT_GRAY60,
        label="Ideal Linear Scaling",
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)

    ax.set_yscale("log", base=2)
    ax.set_yticks(x_ticks)
    ax.set_yticklabels(x_ticks)
    ax.grid(True, which="major", ls="--", lw=0.5)

    if show:
        plt.show()
    else:
        plt.savefig(out_path / "strong_scaling_speedup.png")


# ── CLI ───────────────────────────────────────────────────────────────────────


def collect_paths(path: str) -> list[Path]:
    p = Path(path)
    if p.is_dir():
        return sorted(p.glob("run_*.out"))
    return sorted(Path(f) for f in glob.glob(path))


def main(
    logs_dir: str,
    out_path: str,
    show: bool = False,
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
            records.extend(run)

    if not records:
        print("No valid records parsed.")
        sys.exit(1)

    df = pd.DataFrame.from_records(records)
    df = process_records(df)
    df.info()

    if not out_path:
        csv_out = Path(logs_dir).parent / "results.csv"
    else:
        csv_out = Path(out_path).with_suffix(".csv")

    df.to_csv(csv_out)

    if not out_path:
        plot_out = Path(logs_dir).parent
    else:
        plot_out = Path(out_path)

    plot(df, plot_out, show=show)
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
    parser.add_argument(
        "--show", action="store_true", help="Whether to display the plot interactively."
    )
    parser.add_argument(
        "--out_path",
        default=None,
        help="By default, will choose the parent directory of the logs.",
    )
    args = parser.parse_args()
    main(args.logs_dir, args.out_path, show=args.show)
