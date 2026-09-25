# Range-finder stability experiment

This directory reproduces the figure comparing the residual and loss of
orthogonality of randomized range finder variants. 

It reproduces Fig. 4 in "A GPU-Accelerated Blocked Adaptive Randomized Range Finder Based on an Implicit Householder QR Decomposition" by Carolin Penke and Andreas Herten.

It compares
(a) a variant based on block classical Gram--Schmidt (with and without
reorthogonalization), from "A randomized blocked algorithm for efficiently computing  rank-revealing factorizations of matrices" by Per-Gunnar Martinsson and Sergey Voronin, with and without reorthogonalization,

(b) a Householder block adaptive randomized range finder, proposed in the manuscript.

The experiment uses a deterministic random seed and single-precision
arithmetic.

## Reproduce the figure from the included data

Install the dependencies and render the publication figure:

```console
uv run python plot_rangefinder_orthogonality_loss.py
```

This reads the four included `orthogonality_loss_*.csv` files and writes:

- `rangefinder_orthogonality_loss.pdf`
- `rangefinder_orthogonality_loss.png`

## Reproduce the experiment and figure from scratch

Regenerate the CSV data before rendering the figure:

```console
uv run python experiment_rangefinder_orthogonality_loss.py
uv run python plot_rangefinder_orthogonality_loss.py
```

The experiment constructs a 1024-by-1024 matrix with a prescribed
14-decade singular-value decay, uses blocks of 64 vectors, and runs all three
methods with random seed 1. It overwrites the four CSV files in this directory;
the plotting command then overwrites the PDF and PNG outputs.

To generate the optional self-contained interactive version from the CSV data,
run:

```console
uv run python plot_rangefinder_orthogonality_loss_interactive.py
```

Open `rangefinder_orthogonality_loss_interactive.html` in a web browser to use
the zoom and hover controls. No internet connection is needed to view it.
