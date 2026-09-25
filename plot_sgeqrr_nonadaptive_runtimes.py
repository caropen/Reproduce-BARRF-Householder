#!/usr/bin/env python3

"""Create a two-panel manuscript PDF of randomized-rangefinder runtimes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys


@dataclass(frozen=True)
class SeriesSpec:
    filename: str
    x_column: str
    label: str
    color: str
    marker: str


SERIES = (
    SeriesSpec(
        "cpu.csv", "k", "Non-adaptive, NVPL (CPU)", "#d62728", "D"
    ),
    SeriesSpec(
        "magma_gpu.csv", "k", "Non-adaptive, MAGMA (GPU)", "#1f77b4", "o"
    ),
    SeriesSpec(
        "cusolver_gpu.csv", "k", "Non-adaptive, cuSOLVER (GPU)", "#ff7f0e", "s"
    ),
    SeriesSpec(
        "adaptive_magma_gpu.csv",
        "vcols",
        "Adaptive (MAGMA, GPU), proposed",
        "#2ca02c",
        "^",
    ),
)
OUTPUT_NAME = "rangefinder_runtimes.pdf"
FULL_X_TICKS = (
    # 128,
    # 256,
    1024,
    # 1536,
    2048,
    # 2560,
    3072,
    4096,
    5120,
    6144,
    7168,
    8192,
    9216,
    10240,
    11264,
    12288,
    13312,
    14336,
    15360,
    16384,
)


def parse_arguments() -> argparse.Namespace:
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot CPU, MAGMA, cuSOLVER, and adaptive rangefinder runtimes."
    )
    parser.add_argument(
        "results_directory",
        nargs="?",
        type=Path,
        default=(
            script_directory / "benchmark_results" / "fixed_rank_range_finder"
        ),
        help=(
            "directory containing the four benchmark CSV files "
            "(default: benchmark_results/fixed_rank_range_finder beside this script)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=f"output PDF (default: RESULTS_DIRECTORY/{OUTPUT_NAME})",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="use a logarithmic runtime axis",
    )
    return parser.parse_args()


def import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        import matplotlib.pyplot as plt
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
    return plt


def load_series(csv_file: Path, x_column: str) -> tuple[list[int], list[float]]:
    if not csv_file.is_file():
        raise ValueError(f"missing CSV file: {csv_file}")

    with csv_file.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {x_column, "mean_seconds", "status"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{csv_file.name} is missing columns: {', '.join(sorted(missing))}"
            )

        points: list[tuple[int, float]] = []
        for line_number, row in enumerate(reader, start=2):
            if row["status"] != "ok":
                continue
            try:
                x_value = int(row[x_column])
                runtime = float(row["mean_seconds"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{csv_file.name}:{line_number} has invalid numeric data"
                ) from error
            if x_value <= 0 or runtime <= 0.0:
                raise ValueError(
                    f"{csv_file.name}:{line_number} requires positive values"
                )
            points.append((x_value, runtime))

    if not points:
        raise ValueError(f"{csv_file.name} has no successful runtimes")
    points.sort()
    x_values = [point[0] for point in points]
    if len(x_values) != len(set(x_values)):
        raise ValueError(f"{csv_file.name} has duplicate x-axis values")
    return x_values, [point[1] for point in points]


def main() -> int:
    arguments = parse_arguments()
    plt = import_matplotlib()
    results_directory = arguments.results_directory.resolve()
    output_file = (
        arguments.output.resolve()
        if arguments.output
        else results_directory / OUTPUT_NAME
    )
    if output_file.suffix.lower() != ".pdf":
        print("Error: --output must use the .pdf extension", file=sys.stderr)
        return 2

    try:
        loaded = [
            (spec, *load_series(results_directory / spec.filename, spec.x_column))
            for spec in SERIES
        ]
        expected_x = loaded[0][1]
        if any(x_values != expected_x for _, x_values, _ in loaded[1:]):
            raise ValueError(
                "the benchmark CSV files do not contain the same x-axis values"
            )
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    figure, axis = plt.subplots(
        figsize=(3.5, 2.6), constrained_layout=True
    )
    for spec, x_values, runtimes in loaded:
        axis.plot(
            x_values,
            runtimes,
            label=spec.label,
            color=spec.color,
            marker=spec.marker,
            linewidth=1.25,
            markersize=3.5,
            markeredgewidth=0.5,
        )

    if arguments.log_y:
        axis.set_yscale("log")
        # axis.set_ylim(top=6)
    else:
        axis.set_ylim(0, 6)
    axis.tick_params(axis="both", labelsize=6.5, width=0.6, length=2.5)
    axis.grid(
        True, which="both", linestyle="--", linewidth=0.45, alpha=0.35
    )

    title_suffix = " (dummy data)" if "dummy" in results_directory.name else ""
    axis.set_title(
        f"Householder range finder runtime, $M=N=16\\,384${title_suffix}", fontsize=8.5
    )
    axis.set_xlabel(
        "Number of acquired basis vectors", fontsize=8
    )
    axis.set_ylabel("Mean runtime (s)", fontsize=8)

    ticks = [value for value in FULL_X_TICKS if value in expected_x]
    axis.set_xticks(ticks)
    axis.set_xticklabels(ticks, rotation=45, ha="right")
    axis.set_xlim(0, max(expected_x) * 1.05)
    axis.legend(
        fontsize=6.5,
        handlelength=1.8,
        borderpad=0.4,
        labelspacing=0.3,
        frameon=True,
        loc="lower right",
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, format="pdf")
    plt.close(figure)
    print(f"Created {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
