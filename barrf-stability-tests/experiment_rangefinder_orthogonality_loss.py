"""Compare residuals, stopping criteria, and orthogonality loss.

Compares three methods on the same ill-conditioned, slowly decaying
test matrix:
  1. Block classical Gram-Schmidt (BCGS) range finder with inner Householder
     QR and a single outer projection (Martinsson and Voronin, 2016)
  2. BCGS range finder with inner Householder QR and reorthogonalization
     (Martinsson and Voronin, 2016)
  3. Householder Block Adaptive Randomized Range Finder
     (Penke and Herten, 2026)

Writes one CSV file per method containing the accumulated rank, tracked
stopping quantity, true residual, and orthogonality loss. It also writes
a separate CSV containing the optimal rank-k lower bound sigma_{k+1}(A).

Usage:
    python experiment_rangefinder_orthogonality_loss.py
    python plot_rangefinder_orthogonality_loss.py
"""

import csv

import numpy as np
from scipy.linalg.lapack import sgemqrt, sgeqrt, sorgqr


METHOD_CSV_FILES = {
    "Block CGS":
        "orthogonality_loss_block_cgs_no_reorth.csv",
    "Block CGS (with reorthogonalization)":
        "orthogonality_loss_block_cgs_reorth.csv",
    "Householder BARFF": "orthogonality_loss_householder.csv",
}
OPTIMAL_BOUND_CSV = "orthogonality_loss_optimal_bound.csv"
METHOD_FIELDNAMES = (
    "rank",
    "tracked_stopping_quantity",
    "true_residual",
    "orthogonality_loss",
    "tracked_expensive_2",
    "tracked_expensive_fro",
)


# ----------------------------------------------------------------------
# Test matrix: slowly decaying, deeply ill-conditioned spectrum.
# ----------------------------------------------------------------------

def make_test_matrix(m, n, r, seed=1, kappa_decades=14):
    """Construct the matrix and return its actual and target spectra."""
    rng = np.random.default_rng(seed)
    U_sample = rng.standard_normal((m, r)).astype(np.float32)
    V_sample = rng.standard_normal((n, r)).astype(np.float32)
    Uf, _ = np.linalg.qr(U_sample)
    Vf, _ = np.linalg.qr(V_sample)
    target_s = np.logspace(
        0, -kappa_decades, r, dtype=np.float32
    )
    A = np.asarray((Uf * target_s) @ Vf.T, dtype=np.float32)

    # Rounding the deeply ill-conditioned construction to float32 changes its
    # smallest singular values.  Use the spectrum of the matrix all methods
    # actually receive so the plotted lower bound remains valid.
    singular_values = np.linalg.svd(A, compute_uv=False)
    return A, singular_values, target_s


# ----------------------------------------------------------------------
# Run one method, tracking metrics at every block.
# ----------------------------------------------------------------------


def run_block_cgs(A, b, n_blocks, seed, reorth):
    """Run Algorithm 2 for a fixed number of blocks, tracking each step."""
    rng = np.random.default_rng(seed)
    A_orig = np.asarray(A, dtype=np.float32)
    A_work = A_orig.copy()
    m, n = A_work.shape

    Q = np.empty((m, 0), dtype=A_work.dtype)
    B = np.empty((0, n), dtype=A_work.dtype)

    ranks, tracked, true_resid, orth_loss = [], [], [], []

    for _ in range(n_blocks):
        Omega = rng.standard_normal((n, b)).astype(np.float32)
        Q_new, _ = np.linalg.qr(A_work @ Omega, mode="reduced")

        if reorth:
            Q_new = Q_new - Q @ (Q.T @ Q_new)
            Q_new, _ = np.linalg.qr(Q_new, mode="reduced")

        B_new = Q_new.T @ A_work
        Q = np.hstack([Q, Q_new])
        B = np.vstack([B, B_new])
        A_work = A_work - Q_new @ B_new

        ranks.append(Q.shape[1])
        tracked.append(np.linalg.norm(A_work, 'fro'))
        true_resid.append(
            np.linalg.norm(A_orig - Q @ (Q.T @ A_orig), 2)
        )
        orth_loss.append(
            np.linalg.norm(
                Q.T @ Q - np.eye(Q.shape[1], dtype=np.float32), 2
            )
        )

    return {
        "ranks": ranks,
        "tracked": tracked,
        "true_resid": true_resid,
        "orth_loss": orth_loss,
    }


def _check_lapack_info(routine, info):
    """Raise a useful exception if a low-level LAPACK call failed."""
    if info < 0:
        raise ValueError(
            f"{routine}: argument {-info} had an illegal value"
        )
    if info > 0:
        raise np.linalg.LinAlgError(
            f"{routine} failed with LAPACK info={info}"
        )


def run_householder(A, b, n_blocks, seed, eps=0.0, switcherrorat=5e-3):
    """Run the manuscript's implicit Householder block range finder.

    ``sgeqrt`` stores every sampled panel as a compact-WY reflector and
    ``sgemqrt`` applies its transpose to the active rows of B.  Thus B is
    maintained as Q.T A.  For the experiment's diagnostics, ``sorgqr`` forms
    the current Q explicitly.  ``eps`` is an absolute Frobenius-norm
    tolerance; ``n_blocks`` caps the experiment.
    """

    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError("A must be a two-dimensional array")
    if np.iscomplexobj(A):
        raise TypeError("A must be real because sgeqrt is a real routine")
    if isinstance(b, (bool, np.bool_)) or not isinstance(
            b, (int, np.integer)):
        raise TypeError("b must be an integer")
    if isinstance(n_blocks, (bool, np.bool_)) or not isinstance(
            n_blocks, (int, np.integer)):
        raise TypeError("n_blocks must be an integer")
    b = int(b)
    n_blocks = int(n_blocks)
    if b <= 0:
        raise ValueError("b must be positive")
    if n_blocks < 0:
        raise ValueError("n_blocks must be nonnegative")
    eps = float(eps)
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("eps must be a finite, nonnegative tolerance")
    eps_squared = eps * eps

    m, n = A.shape
    rank_limit = min(m, n, n_blocks * b)
    generator = np.random.default_rng(seed)

    A_work = np.array(A, dtype=np.float32, order="F", copy=True)
    if not np.all(np.isfinite(A_work)):
        raise ValueError("A must contain only finite values")
    B = A_work.copy(order="F")
    remaining_energy_cheap = float(
        np.sum(A_work.astype(np.float64) ** 2)
    )
    remaining_energy_expensive2 = np.inf

    householder_vectors = np.zeros(
        (m, rank_limit), dtype=np.float32, order="F"
    )
    householder_scalars = np.empty(rank_limit, dtype=np.float32)
    ranks, tracked_cheap, tracked_expensive2, tracked_expensive_fro, true_resid, orth_loss = [], [], [], [], [], []

    rank = 0
    # eps=0 is used by this fixed-rank experiment to disable early stopping.
    while rank < rank_limit and (
            eps == 0.0 or max(remaining_energy_cheap, remaining_energy_expensive2 ** 2) > eps_squared):
        width = min(b, rank_limit - rank, m - rank)
        omega = generator.standard_normal((n, width)).astype(np.float32)
        sampled_panel = np.asfortranarray(B[rank:, :] @ omega)

        packed_v, block_t, info = sgeqrt(
            width, sampled_panel, overwrite_a=1
        )
        _check_lapack_info("sgeqrt", info)

        active_rows = np.array(
            B[rank:, :], dtype=np.float32, order="F", copy=True
        )
        active_rows, info = sgemqrt(
            packed_v, block_t, active_rows, side="L", trans="T",
            overwrite_c=1
        )
        _check_lapack_info("sgemqrt", info)
        B[rank:, :] = active_rows

        next_rank = rank + width
        householder_vectors[rank:, rank:next_rank] = packed_v
        householder_scalars[rank:next_rank] = np.diag(block_t)

        new_rows = B[rank:next_rank, :].astype(np.float64)
        remaining_energy_cheap -= float(np.sum(new_rows ** 2))
        remaining_energy_expensive2 = float(np.linalg.norm(B[next_rank:,:],2));
        remaining_energy_expensive_fro = float(np.linalg.norm(B[next_rank:,:],'fro'));
        rank = next_rank

        # Create Q explicitly here to check. 
        Q, _, info = sorgqr(
            householder_vectors[:, :rank], householder_scalars[:rank]
        )
        _check_lapack_info("sorgqr", info)

        ranks.append(rank)
        tracked_cheap.append(np.sqrt(max(remaining_energy_cheap, 0.0)))
        tracked_expensive2.append(remaining_energy_expensive2)
        tracked_expensive_fro.append(remaining_energy_expensive_fro)
        true_resid.append(
            np.linalg.norm(A_work - Q @ (Q.T @ A_work), 2)
        )
        orth_loss.append(
            np.linalg.norm(
                Q.T @ Q - np.eye(rank, dtype=np.float32), 2
            )
        )

    return dict(ranks=ranks, tracked_cheap=tracked_cheap, tracked_expensive2=tracked_expensive2, tracked_expensive_fro=tracked_expensive_fro, true_resid=true_resid,
                orth_loss=orth_loss)


def _format_float(value):
    """Return a round-trip-safe CSV representation of a metric."""
    return "" if value is None else repr(float(value))


def write_method_csv(filename, result):
    """Write one method's metrics, or only the header if unavailable."""
    with open(filename, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=METHOD_FIELDNAMES)
        writer.writeheader()
        if result is None:
            return

        tracked = result.get("tracked", result.get("tracked_cheap"))
        n_rows = len(result["ranks"])
        tracked_2 = result.get("tracked_expensive2", [None] * n_rows)
        tracked_fro = result.get("tracked_expensive_fro", [None] * n_rows)
        for i in range(n_rows):
            writer.writerow({
                "rank": int(result["ranks"][i]),
                "tracked_stopping_quantity": _format_float(tracked[i]),
                "true_residual": _format_float(result["true_resid"][i]),
                "orthogonality_loss": _format_float(result["orth_loss"][i]),
                "tracked_expensive_2": _format_float(tracked_2[i]),
                "tracked_expensive_fro": _format_float(tracked_fro[i]),
            })


def write_optimal_bound_csv(filename, ranks, singular_values,
                            target_singular_values):
    """Write actual and prescribed sigma_(k+1) reference curves."""
    bound_ranks = np.asarray(ranks, dtype=int)
    bound_ranks = bound_ranks[bound_ranks < singular_values.size]
    with open(filename, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file,
                                fieldnames=("rank", "optimal_bound",
                                            "target_bound"))
        writer.writeheader()
        for rank in bound_ranks:
            writer.writerow({
                "rank": int(rank),
                "optimal_bound": _format_float(singular_values[rank]),
                "target_bound": _format_float(
                    target_singular_values[rank]
                ),
            })


# ----------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------

def main():
    m, n, r = 1024, 1024, 1024
    b = 64
    n_blocks = n // b
    seed = 1

    A, singular_values, target_singular_values = make_test_matrix(
        m, n, r, seed=seed
    )
    results = {}
    results["Block CGS"] = run_block_cgs(
        A, b, n_blocks, seed, reorth=False)
    results["Block CGS (with reorthogonalization)"] = run_block_cgs(
        A, b, n_blocks, seed, reorth=True)

    results["Householder BARFF"] = run_householder(
        A, b, n_blocks, seed
    )
    # print(results);

    for name, filename in METHOD_CSV_FILES.items():
        write_method_csv(filename, results[name])
        print(f"Saved {filename}")

    reference_ranks = results["Block CGS (with reorthogonalization)"]["ranks"]
    write_optimal_bound_csv(OPTIMAL_BOUND_CSV, reference_ranks,
                            singular_values, target_singular_values)
    print(f"Saved {OPTIMAL_BOUND_CSV}")

if __name__ == "__main__":
    main()
