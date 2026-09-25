#!/usr/bin/env python3

"""Create a manuscript-ready, three-panel SGEQRR runtime heatmap."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


MATRIX_SIZES = [1024, 2048, 4096, 16384]
BLOCK_SIZES = [64, 128, 256]
IMPLEMENTATIONS = {
    "sgeqrr3_cpu": "CPU",
    "sgeqrr3_gpu_1_queue": "GPU (one queue)",
    "sgeqrr3_gpu": "GPU (two queues)",
}
REQUIRED_COLUMNS = {
    "matrix_size",
    "block_size",
    "lr_tol",
    "warmup_seconds",
    "sample_1_seconds",
    "sample_2_seconds",
    "sample_3_seconds",
    "sample_4_seconds",
    "sample_5_seconds",
    "median_seconds",
    "vcols",
    "status",
}
PANEL_TITLES = {
    "sgeqrr3_cpu": "(a) CPU",
    "sgeqrr3_gpu_1_queue": "(b) GPU\N{EM DASH}one queue",
    "sgeqrr3_gpu": "(c) GPU\N{EM DASH}two queues",
}
OUTPUT_NAME = "sgeqrr3_runtime_heatmaps.pdf"


def parse_arguments() -> argparse.Namespace:
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot the three SGEQRR runtime heatmaps in one PDF figure."
    )
    parser.add_argument(
        "results_directory",
        nargs="?",
        type=Path,
        default=(
            script_directory / "benchmark_results" / "adaptive_range_finder"
        ),
        help=(
            "directory containing the three benchmark CSV files "
            "(default: benchmark_results/adaptive_range_finder beside this script)"
        ),
    )
    return parser.parse_args()


def import_plotting_packages():
    try:
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from matplotlib.patches import Rectangle
        import numpy as np
        import pandas as pd
        import seaborn as sns
    except ModuleNotFoundError as error:
        print(
            f"Missing Python package: {error.name}\n",
            "Run script in uv environment:\n"
            "uv sync --locked\n"
            "uv run --locked plot_randQB_manuscript_heatmaps.py\n"
            "Or install the plotting dependencies with:\n"
            "  python3 -m pip install pandas matplotlib seaborn\n",
            file=sys.stderr,
        )
        raise SystemExit(2) from error

    return plt, LogNorm, Rectangle, np, pd, sns


def load_grid(csv_file: Path, pd, np):
    if not csv_file.is_file():
        raise ValueError(f"missing CSV file: {csv_file}")

    data = pd.read_csv(csv_file)
    missing_columns = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing_columns:
        raise ValueError(
            f"{csv_file.name} is missing columns: {', '.join(missing_columns)}"
        )

    data["matrix_size"] = pd.to_numeric(data["matrix_size"], errors="coerce")
    data["block_size"] = pd.to_numeric(data["block_size"], errors="coerce")
    data["median_seconds"] = pd.to_numeric(
        data["median_seconds"], errors="coerce"
    )
    if data[["matrix_size", "block_size"]].isna().any().any():
        raise ValueError(f"{csv_file.name} has nonnumeric grid values")

    data["matrix_size"] = data["matrix_size"].astype(int)
    data["block_size"] = data["block_size"].astype(int)
    if data.duplicated(["matrix_size", "block_size"]).any():
        raise ValueError(f"{csv_file.name} has duplicate grid entries")

    expected_grid = {
        (matrix_size, block_size)
        for matrix_size in MATRIX_SIZES
        for block_size in BLOCK_SIZES
    }
    actual_grid = set(zip(data["matrix_size"], data["block_size"]))
    if actual_grid != expected_grid:
        raise ValueError(f"{csv_file.name} does not contain the expected 4x3 grid")

    successful = (
        data["status"].eq("ok")
        & data["median_seconds"].notna()
        & data["median_seconds"].gt(0.0)
    )
    data.loc[~successful, "median_seconds"] = np.nan
    return data.pivot(
        index="matrix_size", columns="block_size", values="median_seconds"
    ).reindex(index=MATRIX_SIZES, columns=BLOCK_SIZES)


def annotation_color(value: float, norm, color_map) -> str:
    red, green, blue, _ = color_map(norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#202020" if luminance > 0.58 else "white"


def plot_panel(axis, grid, title, norm, color_map, Rectangle, pd, sns) -> None:
    sns.heatmap(
        grid,
        mask=grid.isna(),
        cmap=color_map,
        norm=norm,
        cbar=False,
        square=True,
        linewidths=0.6,
        linecolor="white",
        xticklabels=BLOCK_SIZES,
        yticklabels=MATRIX_SIZES,
        ax=axis,
    )

    for row_index, matrix_size in enumerate(MATRIX_SIZES):
        for column_index, block_size in enumerate(BLOCK_SIZES):
            value = grid.loc[matrix_size, block_size]
            if pd.isna(value):
                axis.add_patch(
                    Rectangle(
                        (column_index, row_index),
                        1,
                        1,
                        facecolor="#d9d9d9",
                        edgecolor="white",
                        linewidth=0.6,
                    )
                )
                label = "FAIL"
                color = "#8b0000"
                weight = "bold"
            else:
                label = f"{value:.3g}"
                color = annotation_color(value, norm, color_map)
                weight = "normal"

            axis.text(
                column_index + 0.5,
                row_index + 0.5,
                label,
                ha="center",
                va="center",
                color=color,
                fontsize=6.5,
                fontweight=weight,
            )

    axis.set_title(title, fontsize=8.5, pad=5)
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.tick_params(axis="both", labelsize=7, length=0)
    axis.set_xticklabels(BLOCK_SIZES, rotation=0)
    axis.set_yticklabels(MATRIX_SIZES, rotation=0)


def create_figure(grids, output_file, plt, LogNorm, Rectangle, np, pd, sns):
    positive_values = np.concatenate(
        [grid.to_numpy()[np.isfinite(grid.to_numpy())] for grid in grids.values()]
    )
    if positive_values.size == 0:
        raise ValueError("no successful runtimes to plot")

    norm = LogNorm(
        vmin=float(positive_values.min()),
        vmax=float(positive_values.max()),
    )
    color_map = plt.get_cmap("viridis").copy()
    color_map.set_bad("#d9d9d9")

    sns.set_theme(style="white", context="paper")
    figure, axes = plt.subplots(
        1,
        len(IMPLEMENTATIONS),
        figsize=(7.2, 2.8),
        sharey=True,
        constrained_layout=True,
    )

    for axis, implementation in zip(axes, IMPLEMENTATIONS):
        plot_panel(
            axis,
            grids[implementation],
            PANEL_TITLES[implementation],
            norm,
            color_map,
            Rectangle,
            pd,
            sns,
        )

    for axis in axes[1:]:
        axis.tick_params(axis="y", labelleft=False)

    figure.supxlabel("Block size", fontsize=8)
    figure.supylabel("Matrix size", fontsize=8)
    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=color_map)
    scalar_mappable.set_array([])
    color_bar = figure.colorbar(
        scalar_mappable,
        ax=axes,
        location="right",
        fraction=0.035,
        pad=0.02,
    )
    color_bar.set_label("Median runtime (s, log scale)", fontsize=7.5)
    color_bar.ax.tick_params(labelsize=7)

    figure.savefig(output_file, format="pdf")
    plt.close(figure)
    return norm.vmin, norm.vmax


def main() -> int:
    arguments = parse_arguments()
    plt, LogNorm, Rectangle, np, pd, sns = import_plotting_packages()
    results_directory = arguments.results_directory.resolve()
    output_file = results_directory / OUTPUT_NAME

    try:
        grids = {
            implementation: load_grid(
                results_directory / f"{implementation}.csv", pd, np
            )
            for implementation in IMPLEMENTATIONS
        }
        minimum, maximum = create_figure(
            grids, output_file, plt, LogNorm, Rectangle, np, pd, sns
        )
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(f"Created {output_file}")
    print(f"Shared color range: {minimum:.6g} to {maximum:.6g} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
