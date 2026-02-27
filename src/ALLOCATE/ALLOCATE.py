from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from anndata import AnnData



from .Single_adaptive_OT import get_cost
from .Single_adaptive_OT import adaptive_ot_with_KL
from .Spatial_adaptive_OT import fused_gw_adaptive_ot

def scALLOCATE(
    adata: AnnData,
    *,
    early_tp: str,
    late_tp: str,
    time_key: str = "stage",
    rep: str = "X_pca",
    n_dim: int | None = None,
    epsilon: float = 0.1,
    lambda_val: float = 0.5,
    max_iters: int = 100,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    List[Dict],
    List[float],
    List[float],
]:
    """
    Run scALLOCATE: compute OT-based allocation between two time points.

    This function:
    1) Computes the OT cost matrix from a low-dimensional embedding.
    2) Solves an entropic OT problem with KL regularization.
    3) Returns the optimized marginal, transport plan, and diagnostics.

    Parameters
    ----------
    adata
        AnnData object containing single-cell data.
    early_tp
        Label of the early time point in ``adata.obs[time_key]``.
    late_tp
        Label of the late time point in ``adata.obs[time_key]``.
    time_key
        Key in ``adata.obs`` indicating developmental stage or time.
    rep
        Key in ``adata.obsm`` specifying the embedding used for cost construction.
    n_dim
        Number of embedding dimensions to use. If ``None``, all dimensions are used.
    epsilon
        Entropic regularization parameter.
    lambda_val
        KL regularization weight.
    max_iters
        Maximum number of outer iterations in the OT solver.

    Returns
    -------
    a
        Optimized source marginal distribution.
    pi
        Optimal transport plan between early and late cells.
    stats
        Per-iteration statistics from the solver.
    a_history
        History of marginal updates.
    pi_history
        History of transport plan updates.
    """
    # --- compute cost matrix ---
    C = get_cost(
        adata,
        early_tp=early_tp,
        late_tp=late_tp,
        time_key=time_key,
        rep=rep,
        n_dim=n_dim,
    )

    # --- solve OT problem ---
    a, pi, stats, a_history, pi_history = adaptive_ot_with_KL(
        C,
        epsilon=epsilon,
        lambda_val=lambda_val,
        max_iters=max_iters,
    )

    return a, pi, stats, a_history, pi_history





def stALLOCATE(
    C: np.ndarray,
    D: np.ndarray,
    D_prime: np.ndarray,
    *,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    lambda_val: float = 0.5,
    beta: float = 0.9,
    max_outer: int = 40,
    max_middle: int = 100,
    max_inner: int = 1000,
    tol: float = 1e-4,
    sinkhorn_tol: float = 1e-4,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], List[np.ndarray]]:
    """
    Run fused GW + adaptive OT allocation (three-level optimization).

    This function is a high-level wrapper around `fused_gw_adaptive_ot`.
    It combines:
      - inter-domain feature cost `C` (n x m),
      - intra-domain structure costs `D` (n x n) and `D_prime` (m x m),
    and solves a fused objective using entropic OT with KL regularization.

    Parameters
    ----------
    C
        Feature cost matrix (n, m). Typically derived from gene expression / embeddings.
    D
        Intra-domain distance matrix for source (n, n). Usually spatial distance or graph distance.
    D_prime
        Intra-domain distance matrix for target (m, m). Usually spatial distance or graph distance.
    alpha
        Weight of structure term relative to feature cost.
        - alpha = 0 → pure feature OT
        - alpha = 1 → pure structure (GW-like)
    epsilon
        Entropic regularization strength:
        - larger epsilon → smoother / denser pi
        - smaller epsilon → sharper / sparser pi (but more numerical instability)
    lambda_val
        KL penalty weight between row marginal of pi and source distribution a.
    beta
        Balance between standard GW term0 and triplet-style term2 in the structural term.
        - beta = 1 → standard GW-like term0 only
        - beta = 0 → triplet term2 only
    max_outer, max_middle, max_inner
        Iteration limits for the three-level optimization.
    tol
        Convergence tolerance for outer & middle loops (loss change).
    sinkhorn_tol
        Convergence tolerance for inner Sinkhorn loop (dual change).
    verbose
        If True, print progress messages.

    Returns
    -------
    a
        Final source marginal distribution (n,).
    pi
        Final transport plan (n, m).
    stats
        Statistics recorded per outer iteration (loss trajectories, entropy, etc.).
    a_history
        History of the source marginal `a` across outer iterations.
    """
    # ---- input sanity checks ----
    if C.ndim != 2:
        raise ValueError(f"`C` must be 2D, got shape {C.shape}.")
    if D.ndim != 2 or D_prime.ndim != 2:
        raise ValueError("`D` and `D_prime` must be 2D square distance matrices.")
    if D.shape[0] != D.shape[1]:
        raise ValueError(f"`D` must be square, got shape {D.shape}.")
    if D_prime.shape[0] != D_prime.shape[1]:
        raise ValueError(f"`D_prime` must be square, got shape {D_prime.shape}.")
    if C.shape[0] != D.shape[0]:
        raise ValueError("C.shape[0] must match D.shape[0] (source size).")
    if C.shape[1] != D_prime.shape[0]:
        raise ValueError("C.shape[1] must match D_prime.shape[0] (target size).")

    # ---- run solver ----
    if verbose:
        print(
            f"[scALLOCATE_fusedGW] Running fused GW-OT with "
            f"alpha={alpha}, beta={beta}, epsilon={epsilon}, lambda={lambda_val}"
        )

    a, pi, stats, a_history = fused_gw_adaptive_ot(
        C=C,
        D=D,
        D_prime=D_prime,
        alpha=alpha,
        epsilon=epsilon,
        lambda_val=lambda_val,
        beta=beta,
        max_outer=max_outer,
        max_middle=max_middle,
        max_inner=max_inner,
        tol=tol,
        sinkhorn_tol=sinkhorn_tol,
    )

    if verbose:
        print(
            f"[scALLOCATE_fusedGW] Done. "
            f"outer_iters={len(stats)}, pi_norm={np.linalg.norm(pi):.4f}"
        )

    return a, pi, stats, a_history
