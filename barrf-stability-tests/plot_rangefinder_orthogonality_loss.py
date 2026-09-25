"""Plot residual stall and orthogonality collapse from experiment CSVs.

This script performs no experiment computations. It reads the fixed CSV
files produced by experiment_rangefinder_orthogonality_loss.py and writes manuscript-
ready PDF and PNG versions of the two-panel figure.

Usage:
    python experiment_rangefinder_orthogonality_loss.py
    python plot_rangefinder_orthogonality_loss.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
OPTIMAL_BOUND_CSV = "orthogonality_loss_optimal_bound.csv"
OUTPUT_PNG = "rangefinder_orthogonality_loss.png"
OUTPUT_PDF = "rangefinder_orthogonality_loss.pdf"
METHOD_FIELDNAMES = (
    "rank",
    "tracked_stopping_quantity",
    "true_residual",
    "orthogonality_loss",
    "tracked_expensive_2",
    "tracked_expensive_fro",
)

TARGET_COLOR = "#CC79A7"
HH_FRO_COLOR = "#E69F00"
EPSILON_COLOR = "#777777"
LINE_WIDTH = 1.2
MARKER_SIZE = 3.5


matplotlib.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _open_csv(filename, expected_fields):
    path = SCRIPT_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing experiment output: {path}. Run "
            "experiment_rangefinder_orthogonality_loss.py first."
        )

    csv_file = path.open(newline="", encoding="utf-8")
    reader = csv.DictReader(csv_file)
    if tuple(reader.fieldnames or ()) != tuple(expected_fields):
        csv_file.close()
        raise ValueError(
            f"Unexpected columns in {filename}: {reader.fieldnames}; "
            f"expected {list(expected_fields)}"
        )
    return csv_file, reader


def _positive_plot_data(ranks, values, description):
    """Return finite positive values suitable for a logarithmic axis."""
    if ranks.shape != values.shape:
        raise ValueError(
            f"Mismatched rank and value counts for {description}: "
            f"{ranks.size} and {values.size}."
        )

    mask = np.isfinite(values) & (values > 0)
    if not np.any(mask):
        raise ValueError(
            f"No positive finite values available for {description}."
        )
    return ranks[mask], values[mask]


def read_method_csv(filename):
    """Load one method CSV; a header-only file returns empty arrays."""
    csv_file, reader = _open_csv(filename, METHOD_FIELDNAMES)
    rows = list(reader)
    csv_file.close()

    result = {
        "ranks": np.asarray([int(row["rank"]) for row in rows], dtype=int),
        "tracked": np.asarray([
            float(row["tracked_stopping_quantity"])
            if row["tracked_stopping_quantity"] else np.nan
            for row in rows
        ]),
        "true_resid": np.asarray([
            float(row["true_residual"]) for row in rows
        ]),
        "orth_loss": np.asarray([
            float(row["orthogonality_loss"]) for row in rows
        ]),
        "tracked_2": np.asarray([
            float(row["tracked_expensive_2"])
            if row["tracked_expensive_2"] else np.nan
            for row in rows
        ]),
        "tracked_fro": np.asarray([
            float(row["tracked_expensive_fro"])
            if row["tracked_expensive_fro"] else np.nan
            for row in rows
        ]),
    }
    return result


def _split_at_switch(ranks, cheap, expensive, switch_at):
    """Split tracked quantities at the switch from cheap to expensive.

    The cheap criterion is used up to and including the first rank where it
    drops below ``switch_at``; from that rank on the expensive criterion is
    used. Both are returned for the switch rank itself. If the cheap
    criterion never drops below ``switch_at``, the expensive part is empty.
    """
    below = np.flatnonzero(np.isfinite(cheap) & (cheap < switch_at))
    if below.size == 0:
        return (ranks, cheap), (ranks[:0], expensive[:0])
    switch = below[0]
    return (
        (ranks[:switch + 1], cheap[:switch + 1]),
        (ranks[switch:], expensive[switch:]),
    )


def read_optimal_bound_csv(filename):
    """Load actual and prescribed rank-k reference bounds."""
    csv_file, reader = _open_csv(
        filename, ("rank", "optimal_bound", "target_bound")
    )
    rows = list(reader)
    csv_file.close()
    return (
        np.asarray([int(row["rank"]) for row in rows], dtype=int),
        np.asarray([float(row["optimal_bound"]) for row in rows]),
        np.asarray([float(row["target_bound"]) for row in rows]),
    )


def main():
    SwitchCriterionAt = 5e-3;

    block_cgs = read_method_csv(
        "orthogonality_loss_block_cgs_no_reorth.csv"
    )
    block_cgs_reorth = read_method_csv(
        "orthogonality_loss_block_cgs_reorth.csv"
    )
    householder = read_method_csv(
        "orthogonality_loss_householder.csv"
    )
    bound_ranks, optimal_bound, target_bound = read_optimal_bound_csv(
        OPTIMAL_BOUND_CSV
    )

    empty_methods = []
    if block_cgs["ranks"].size == 0:
        empty_methods.append("Block CGS")
    if block_cgs_reorth["ranks"].size == 0:
        empty_methods.append("Block CGS (with reorthogonalization)")
    if householder["ranks"].size == 0:
        empty_methods.append("Householder (proposed)")
    if empty_methods:
        raise ValueError(
            "No experiment rows found for " + ", ".join(empty_methods)
            + ". Run experiment_rangefinder_orthogonality_loss.py first."
        )

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.16, 3.6), sharex=True
    )
    fig.subplots_adjust(
        left=0.085, right=0.98, bottom=0.17, top=0.70, wspace=0.30
    )

    actual_ranks, actual_bound = _positive_plot_data(
        bound_ranks, optimal_bound, "realized best rank-k error"
    )
    target_ranks, target_spectrum = _positive_plot_data(
        bound_ranks, target_bound, "prescribed target spectrum"
    )

    ax1.semilogy(
        actual_ranks, actual_bound, color="black", ls=":",
        lw=LINE_WIDTH, zorder=2,
    )
    ax1.semilogy(
        target_ranks, target_spectrum, color=TARGET_COLOR, ls="-.",
        lw=LINE_WIDTH, zorder=2,
    )

    # Block CGS: residual, tracked quantity, and orthogonality loss.
    block_cgs_residual_ranks, block_cgs_residual = _positive_plot_data(
        block_cgs["ranks"], block_cgs["true_resid"],
        "Block CGS residual",
    )
    ax1.semilogy(
        block_cgs_residual_ranks, block_cgs_residual,
        marker="o", ms=MARKER_SIZE, color="#0072B2",
        lw=LINE_WIDTH, zorder=3,
    )
    block_cgs_tracked_ranks, block_cgs_tracked = _positive_plot_data(
        block_cgs["ranks"], block_cgs["tracked"],
        "Block CGS tracked quantity",
    )
    ax1.semilogy(
        block_cgs_tracked_ranks, block_cgs_tracked,
        marker="o", ms=MARKER_SIZE, mfc="white", mew=0.9,
        ls="--", color="#0072B2", lw=LINE_WIDTH,
        alpha=0.85, zorder=3,
    )
    block_cgs_orth_ranks, block_cgs_orth_loss = _positive_plot_data(
        block_cgs["ranks"], block_cgs["orth_loss"],
        "Block CGS orthogonality loss",
    )
    ax2.semilogy(
        block_cgs_orth_ranks, block_cgs_orth_loss,
        marker="o", ms=MARKER_SIZE, color="#0072B2",
        lw=LINE_WIDTH, zorder=3,
    )

    # Block CGS with reorthogonalization: all three plotted quantities.
    reorth_residual_ranks, reorth_residual = _positive_plot_data(
        block_cgs_reorth["ranks"], block_cgs_reorth["true_resid"],
        "Block CGS (with reorthogonalization) residual",
    )
    ax1.semilogy(
        reorth_residual_ranks, reorth_residual,
        marker="s", ms=MARKER_SIZE, color="#009E73",
        lw=LINE_WIDTH, zorder=3,
    )
    reorth_tracked_ranks, reorth_tracked = _positive_plot_data(
        block_cgs_reorth["ranks"], block_cgs_reorth["tracked"],
        "Block CGS (with reorthogonalization) tracked quantity",
    )
    ax1.semilogy(
        reorth_tracked_ranks, reorth_tracked,
        marker="s", ms=MARKER_SIZE, mfc="white", mew=0.9,
        ls="--", color="#009E73", lw=LINE_WIDTH,
        alpha=0.85, zorder=3,
    )
    reorth_orth_ranks, reorth_orth_loss = _positive_plot_data(
        block_cgs_reorth["ranks"], block_cgs_reorth["orth_loss"],
        "Block CGS (with reorthogonalization) orthogonality loss",
    )
    ax2.semilogy(
        reorth_orth_ranks, reorth_orth_loss,
        marker="s", ms=MARKER_SIZE, color="#009E73",
        lw=LINE_WIDTH, zorder=3,
    )

    # Householder (proposed): all three plotted quantities.
    householder_residual_ranks, householder_residual = _positive_plot_data(
        householder["ranks"], householder["true_resid"],
        "Householder (proposed) residual",
    )
    ax1.semilogy(
        householder_residual_ranks, householder_residual,
        marker="^", ms=MARKER_SIZE, color="#D55E00",
        lw=LINE_WIDTH, zorder=3,
    )
    # Tracked quantity: cheap criterion while it is large, then the
    # expensive Frobenius-norm criterion once the cheap one drops below
    # SwitchCriterionAt (both are shown at the switch rank).
    (cheap_ranks, cheap_values), (fro_ranks, fro_values) = _split_at_switch(
        householder["ranks"], householder["tracked"],
        householder["tracked_fro"], SwitchCriterionAt,
    )
    householder_tracked_ranks, householder_tracked = _positive_plot_data(
        cheap_ranks, cheap_values,
        "Householder (proposed) cheap tracked quantity",
    )
    ax1.semilogy(
        householder_tracked_ranks, householder_tracked,
        marker="^", ms=MARKER_SIZE, mfc="white", mew=0.9,
        ls="--", color="#D55E00", lw=LINE_WIDTH,
        alpha=0.85, zorder=3,
    )
    if fro_ranks.size > 0:
        householder_fro_ranks, householder_fro = _positive_plot_data(
            fro_ranks, fro_values,
            "Householder (proposed) Frobenius tracked quantity",
        )
        ax1.semilogy(
            householder_fro_ranks, householder_fro,
            marker="v", ms=MARKER_SIZE, mfc="white", mew=0.9,
            ls="-.", color=HH_FRO_COLOR, lw=LINE_WIDTH,
            alpha=0.85, zorder=3,
        )
    householder_orth_ranks, householder_orth_loss = _positive_plot_data(
        householder["ranks"], householder["orth_loss"],
        "Householder (proposed) orthogonality loss",
    )
    ax2.semilogy(
        householder_orth_ranks, householder_orth_loss,
        marker="^", ms=MARKER_SIZE, color="#D55E00",
        lw=LINE_WIDTH, zorder=3,
    )

    epsilon = np.finfo(np.float32).eps
    for axis, epsilon_x, epsilon_alignment in (
        (ax1, 0.02, "left"),
        (ax2, 0.98, "right"),
    ):
        axis.axhline(
            epsilon, color=EPSILON_COLOR, ls=":", lw=0.9, zorder=1
        )
        axis.annotate(
            r"$\epsilon_{\mathrm{FP32}}$",
            xy=(epsilon_x, epsilon),
            xycoords=("axes fraction", "data"),
            xytext=(0, 2), textcoords="offset points",
            ha=epsilon_alignment, va="bottom", fontsize=6.5,
            color=EPSILON_COLOR,
        )
        axis.grid(
            visible=True, which="major", axis="both",
            color="#D0D0D0", lw=0.45, alpha=0.65,
        )
        axis.set_axisbelow(True)
        axis.tick_params(width=0.8, length=3)

    max_rank = max(
        int(block_cgs["ranks"].max()),
        int(block_cgs_reorth["ranks"].max()),
        int(householder["ranks"].max()),
    )
    for axis in (ax1, ax2):
        axis.set_xlim(0, max_rank)
        axis.set_xticks(np.arange(128, max_rank + 1, 128))

    ax1.set_title("(a) Residual and stopping quantities", pad=4)
    ax1.set_ylabel("Residual")
    ax2.set_title("(b) Loss of orthogonality", pad=4)
    ax2.set_ylabel(r"$\Vert Q^TQ-I\Vert_2$")
    fig.supxlabel(r"Number of acquired basis vectors $k$", y=0.035, fontsize=8)

    # Custom legend for the residual and tracked-quantity panel.
    left_legend_handles = [
        # Block CGS.
        Line2D(
            [], [], color="#0072B2", marker="o",
            lw=LINE_WIDTH, ms=MARKER_SIZE,
            label=r"BCGS Range Finder: $\Vert A-QQ^T A\Vert_2$ (actual)",
        ),
        Line2D(
            [], [], color="#0072B2", marker="o",
            mfc="white", mew=0.9, ls="--", lw=LINE_WIDTH,
            ms=MARKER_SIZE,
            label=r"BCGS Range Finder: $\Vert A-QB\Vert_F$ (tracked)",
        ),
        # Block CGS with reorthogonalization.
        Line2D(
            [], [], color="#009E73", marker="s",
            lw=LINE_WIDTH, ms=MARKER_SIZE,
            label=r"BCGS Range Finder + Reorth.: $\Vert A-QQ^T A\Vert_2$ (actual)",
        ),
        Line2D(
            [], [], color="#009E73", marker="s",
            mfc="white", mew=0.9, ls="--", lw=LINE_WIDTH,
            ms=MARKER_SIZE,
            label=r"BCGS Range Finder + Reorth.: $\Vert A-QB\Vert_F$ (tracked)",
        ),
        # Householder (proposed).
        Line2D(
            [], [], color="#D55E00", marker="^",
            lw=LINE_WIDTH, ms=MARKER_SIZE,
            label=r"HH Range Finder: $\Vert A-QB\Vert_2$ (actual)",
        ),
        Line2D(
            [], [], color="#D55E00", marker="^",
            mfc="white", mew=0.9, ls="--", lw=LINE_WIDTH,
            ms=MARKER_SIZE,
            label=(
                "HH Range Finder:\n"
                r"$(\Vert A\Vert_F^2-\Sigma_{i}\Vert B_i\Vert_F^2)^\frac{1}{2}$ (tracked)"
            ),
        ),
        Line2D(
            [], [], color=HH_FRO_COLOR, marker="v",
            mfc="white", mew=0.9, ls="-.", lw=LINE_WIDTH,
            ms=MARKER_SIZE,
            label=(
                "HH Range Finder:\n"
                r"$\Vert B_{i+1:j}\Vert_F$ (tracked)"
            ),
        ),
        # Reference bounds.
        Line2D(
            [], [], color="black", ls=":", lw=LINE_WIDTH,
            label=r"Actual $\sigma_{k+1}(A)$",
        ),
        Line2D(
            [], [], color=TARGET_COLOR, ls="-.", lw=LINE_WIDTH,
            label=r"Prescribed $\sigma_{k+1}(A)$",
        ),
    ]

    # Custom legend for the orthogonality-loss panel.
    right_legend_handles = [
        Line2D(
            [], [], color="#0072B2", marker="o",
            lw=LINE_WIDTH, ms=MARKER_SIZE,
            label="BCGS Range Finder",
        ),
        Line2D(
            [], [], color="#009E73", marker="s",
            lw=LINE_WIDTH, ms=MARKER_SIZE,
            label="BCGS Range Finder + Reorth.",
        ),
        Line2D(
            [], [], color="#D55E00", marker="^",
            lw=LINE_WIDTH, ms=MARKER_SIZE,
            label="HH Range Finder",
        ),
    ]

    ax1.legend(
        handles=left_legend_handles, loc="center",
        bbox_to_anchor=(0.66, 1.34), ncol=2, frameon=False,
        borderaxespad=0, handlelength=2.2, columnspacing=1.0,
        labelspacing=0.35,
    )
    ax2.legend(
        handles=right_legend_handles, loc="center",
        bbox_to_anchor=(0.5, 1.34), ncol=1, frameon=False,
        borderaxespad=0, handlelength=2.2, labelspacing=0.35,
    )

    output_png = SCRIPT_DIR / OUTPUT_PNG
    output_pdf = SCRIPT_DIR / OUTPUT_PDF
    fig.savefig(output_png, dpi=300)
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Saved {output_png}")
    print(f"Saved {output_pdf}")


if __name__ == "__main__":
    main()
