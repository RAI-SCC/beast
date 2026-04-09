#!/usr/bin/env python3
"""
parse_scaling.py  –  Parse weak-scaling log files and plot avg TFLOPs/s vs Forward TFLOPs.

Usage:
    python parse_scaling.py --logs_dir /path/to/logs/
    python parse_scaling.py --logs_dir /path/to/logs/ --out_path results/plot.png --sweep_id my_sweep
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── KIT Color Palette ─────────────────────────────────────────────────────────
KIT_GREEN = (0 / 255, 150 / 255, 130 / 255)
KIT_BLUE = (70 / 255, 100 / 255, 170 / 255)
KIT_RED = (162 / 255, 34 / 255, 35 / 255)
KIT_ORANGE = (223 / 255, 155 / 255, 27 / 255)
KIT_LIGHTGREEN = (140 / 255, 182 / 255, 60 / 255)
KIT_PURPLE = (163 / 255, 16 / 255, 124 / 255)
KIT_CYAN = (35 / 255, 161 / 255, 224 / 255)
KIT_BROWN = (167 / 255, 130 / 255, 46 / 255)

KIT_GRAY10 = (0.9, 0.9, 0.9)
KIT_GRAY30 = (0.7, 0.7, 0.7)
KIT_GRAY60 = (0.4, 0.4, 0.4)

KIT_PALETTE = [
    KIT_BLUE,
    KIT_GREEN,
    KIT_ORANGE,
    KIT_RED,
    KIT_CYAN,
    KIT_PURPLE,
    KIT_LIGHTGREEN,
    KIT_BROWN,
]


def plot_timings_by_step(df: pd.DataFrame, out_path: Path):

    # I want a bar plot, each color should represent the the timing type, x-axis the step # number, and the y-axis the duration in seconds averaged over the different epochs and ranks.
    width = 0.2
    multiplier = 0

    fix, ax = plt.subplots(figsize=(10, 6))
    timing_types = df["timing"].unique()
    colors = KIT_PALETTE[: len(timing_types)]

    for timing, color in zip(timing_types, colors):
        offset = width * multiplier
        timing_df = df[df["timing"] == timing]
        avg_timing = timing_df.groupby("#batch")["value"].mean().reset_index()
        std_timing = timing_df.groupby("#batch")["value"].std().reset_index()

        ax.bar(
            x=avg_timing["#batch"] + offset,
            height=avg_timing["value"],
            width=width,
            color=color,
            label=timing,
        )
        ax.errorbar(
            x=avg_timing["#batch"] + offset,
            y=avg_timing["value"],
            yerr=std_timing["value"],
            fmt="none",
            ecolor="black",
            capsize=5,
        )
        multiplier += 1

    plt.xticks(np.arange(0, len(avg_timing), 10))
    plt.xlabel("Step #")
    plt.ylabel("Duration (s)")
    plt.title("Average Section Durations per Step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "timings_by_step.png")
    plt.close()


def plot_step_durations(df: pd.DataFrame, out_path: Path):

    # I want a line plot, x-axis the step # number, and the y-axis the duration in seconds averaged over the different epochs and ranks. Each line should represent a different timing type.
    _, ax = plt.subplots(figsize=(10, 6))

    timing_df = df[df["timing"] == "step duration"]
    avg_timing = timing_df.groupby("Step #")["value"].mean().reset_index()
    std_timing = timing_df.groupby("Step #")["value"].std().reset_index()

    ax.plot(avg_timing["Step #"], avg_timing["value"], label="Step Duration")
    ax.errorbar(
        x=avg_timing["Step #"],
        y=avg_timing["value"],
        yerr=std_timing["value"],
        fmt="none",
        ecolor="black",
        capsize=5,
    )

    plt.xlabel("Step #")
    plt.ylabel("Duration (s)")
    plt.title("Average Full Step Durations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "step_durations_plot.png")
    plt.close()


def plot_timings_by_rank(df: pd.DataFrame, out_path: Path):

    # I want a bar plot, each color should represent the the timing type, x-axis the step # number, and the y-axis the duration in seconds averaged over the different epochs and ranks.
    width = 0.2
    multiplier = 0

    _, ax = plt.subplots(figsize=(10, 6))
    timing_types = df["timing"].unique()
    colors = KIT_PALETTE[: len(timing_types)]

    for timing, color in zip(timing_types, colors):
        offset = width * multiplier
        timing_df = df[df["timing"] == timing]
        avg_timing = timing_df.groupby("rank")["value"].mean().reset_index()

        ax.bar(
            x=avg_timing["rank"] + offset,
            height=avg_timing["value"],
            width=width,
            color=color,
            label=timing,
        )
        multiplier += 1

    plt.xlabel("Rank")
    plt.ylabel("Duration (s)")
    plt.title("Average Timing Durations by Rank")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "timings_by_rank.png")
    plt.close()


def get_timings_dataframe(output_path: Path) -> pd.DataFrame:

    dfs = []
    for file in output_path.glob("timings/*.csv"):
        rank = file.name.split("_")[1][4:]
        df = pd.read_csv(file, sep=";")
        df["rank"] = rank
        dfs.append(df)

    steps_per_epoch = dfs[0]["#batch"].max() + 1
    system_df = pd.concat(dfs, ignore_index=True)
    system_df["Step #"] = (system_df["#epoch"] * steps_per_epoch) + system_df["#batch"]
    system_df["step duration"] = (system_df["step_end"] - system_df["step_start"]) / 1e9
    system_df["data loading duration"] = (
        system_df["data_loading_end"] - system_df["data_loading_start"]
    ) / 1e9
    system_df["to_device duration"] = (
        system_df["gpu_time"] - system_df["data_loading_end"]
    ) / 1e9
    system_df["forward/backward duration"] = (
        system_df["step_end"] - system_df["gpu_time"]
    ) / 1e9
    system_df.drop(
        columns=[
            "step_start",
            "data_loading_start",
            "data_loading_end",
            "gpu_time",
            "step_end",
        ],
        inplace=True,
    )
    system_df = system_df.melt(
        id_vars=["rank", "#epoch", "#batch", "Step #"],
        var_name="timing",
        value_name="value",
    )
    system_df["rank"] = system_df["rank"].astype(int)
    return system_df


def main(
    path_str: str,
):

    path = Path(path_str)
    if not path.exists():
        print(f"Error: Path '{path}' does not exist.")
        sys.exit(1)

    run_id = path.name
    partition = path.parent.name
    print(f"Parsing logs in run: {run_id}, partition: {partition} (from path: {path})")

    df = get_timings_dataframe(path)

    plot_timings_by_step(df, path)
    plot_step_durations(df, path)
    plot_timings_by_rank(df, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse weak-scaling .out logs and produce a throughput plot."
    )
    parser.add_argument(
        "path",
        help="Output directory of the run with timings folder",
    )
    args = parser.parse_args()
    main(args.path)
