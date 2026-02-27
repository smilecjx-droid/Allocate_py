'''
An Optimal-Transport based approach for quantifying the spatiotemporal growth of tissues.
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from scipy.spatial.distance import cdist
from anndata import AnnData
from typing import Optional
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap


def get_cost(
    adata: AnnData,
    *,
    early_tp: str = "t1",
    late_tp: str = "t2",
    time_key: str = "time",
    rep: str = "X_pca",
    n_dim: Optional[int] = None,
) -> np.ndarray:
    """
    Compute the pairwise cost (distance) matrix between early and late cells
    based on a low-dimensional embedding.

    Parameters
    ----------
    adata
        AnnData object containing single-cell data.
    early_tp
        Label in `adata.obs[time_key]` corresponding to the early time point.
    late_tp
        Label in `adata.obs[time_key]` corresponding to the late time point.
    time_key
        Key in `adata.obs` indicating the time or condition annotation.
    rep
        Key in `adata.obsm` specifying the embedding to use (e.g., ``"X_pca"``).
    n_dim
        Number of embedding dimensions to use. If ``None``, all available
        dimensions in ``rep`` are used.

    Returns
    -------
    np.ndarray
        A 2D array of shape ``(n_early_cells, n_late_cells)`` containing
        pairwise Euclidean distances.

    Raises
    ------
    KeyError
        If ``rep`` is not found in ``adata.obsm`` or ``time_key`` not in ``adata.obs``.
    ValueError
        If ``n_dim`` exceeds the available embedding dimensionality.
    """
    # --- retrieve embedding ---
    if rep not in adata.obsm:
        raise KeyError(f"Embedding `{rep}` not found in `adata.obsm`.")

    if time_key not in adata.obs:
        raise KeyError(f"Key `{time_key}` not found in `adata.obs`.")

    embedding = np.asarray(adata.obsm[rep], dtype=float)

    # --- handle dimensionality ---
    if n_dim is None:
        n_dim = embedding.shape[1]
    elif n_dim > embedding.shape[1]:
        raise ValueError(
            f"Requested n_dim={n_dim}, but `{rep}` only has "
            f"{embedding.shape[1]} dimensions."
        )

    embedding = embedding[:, :n_dim]

    # --- select early and late cells ---
    mask_early = adata.obs[time_key] == early_tp
    mask_late = adata.obs[time_key] == late_tp

    embedding_early = embedding[mask_early]
    embedding_late = embedding[mask_late]

    # --- compute pairwise distances ---
    cost_matrix = cdist(
        embedding_early,
        embedding_late,
        metric="euclidean",
    )

    return cost_matrix



def compute_dual_loss(
    f: np.ndarray,
    K: np.ndarray,
    g: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    epsilon: float,
    lambda_val: float,
) -> float:
    """
    Compute the dual objective (as written in the original code).

    Notes
    -----
    This matches your original formula:

        term1 = -epsilon * <f, K g>
        term2 = -lambda * sum(a * f^(-epsilon/lambda))
        term3 = lambda*sum(a) + epsilon*sum(log(g)*b)

    Returns
    -------
    float
        Dual loss value.
    """
    # Guard against invalid logs/powers
    g_safe = np.maximum(g, 1e-300)
    f_safe = np.maximum(f, 1e-300)

    term1 = -epsilon * float(np.dot(f_safe, K @ g_safe))
    term2 = -lambda_val * float(np.sum(a * np.power(f_safe, -epsilon / lambda_val)))
    term3 = lambda_val * float(np.sum(a)) + epsilon * float(np.sum(np.log(g_safe) * b))
    return term1 + term2 + term3


def compute_loss(
    pi: np.ndarray,
    C: np.ndarray,
    epsilon: float,
    lambda_val: float,
    a: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Compute primal components given a transport plan pi.

    Returns
    -------
    transport_cost : float
        sum_{ij} pi_ij * C_ij
    neg_entropy : float
        - epsilon * sum_{ij} pi_ij * (log(pi_ij) - 1)  (matches your sign convention)
    kl : float
        lambda * KL(pi 1 || a) in the specific expanded form you used
    total : float
        transport + entropy + kl   (matches your original)
    """
    pi_safe = np.maximum(pi, 1e-300)  # avoid log(0)

    transport = float(np.sum(pi_safe * C))

    # entropy term (your function returns -entropy as a positive-ish monitoring term)
    entropy = epsilon * float(np.sum(pi_safe * (np.log(pi_safe) - 1.0)))

    # marginal on rows
    pi_marginal = np.sum(pi_safe, axis=1)
    pi_marginal_safe = np.maximum(pi_marginal, 1e-300)
    a_safe = np.maximum(a, 1e-300)

    kl = lambda_val * float(
        np.dot(pi_marginal_safe, np.log(pi_marginal_safe) - np.log(a_safe))
        - np.sum(pi_marginal_safe)
        + np.sum(a_safe)
    )

    total_loss = transport + entropy + kl
    return transport, -entropy, kl, total_loss


def adaptive_ot_with_KL(
    C: np.ndarray,
    epsilon: float,
    lambda_val: float,
    max_iters: int = 100,
    tol: float = 1e-6,
    sinkhorn_tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], List[float], List[float]]:
    """
    Adaptive OT with KL regularization (outer update on `a`) + inner Sinkhorn-like updates.

    Parameters
    ----------
    C
        Cost matrix of shape (n, m).
    epsilon
        Entropic regularization strength (> 0).
    lambda_val
        KL regularization weight (> 0).
    max_iters
        Max number of outer iterations.
    tol
        Outer stopping tolerance based on change in total_loss.
    sinkhorn_tol
        Inner stopping tolerance based on change in dual loss (your original criterion).

    Returns
    -------
    a : np.ndarray
        Updated source marginal of shape (n,).
    pi : np.ndarray
        Transport plan of shape (n, m).
    stats : list[dict]
        Per-outer-iter stats including inner stats.
    a_history : list[float]
        Norm of (a_new - a) over outer iterations.
    pi_history : list[float]
        Norm of (pi - pi_prev) over outer iterations (starts from 2nd outer iter).
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if lambda_val <= 0:
        raise ValueError(f"lambda_val must be > 0, got {lambda_val}")
    if C.ndim != 2:
        raise ValueError(f"C must be 2D, got shape {C.shape}")

    n, m = C.shape

    # Construct kernel K = exp(-C/epsilon).
    # Warning: this can underflow/overflow for extreme values.
    # For typical OT cost scales, this is fine; otherwise consider log-domain Sinkhorn.
    K = np.exp(-C / epsilon)
    K_T = K.T

    # Uniform target marginal b, and initialize a uniformly.
    b = np.ones(m, dtype=float) / float(m)
    a = np.ones(n, dtype=float) / float(n)

    stats: List[Dict[str, Any]] = []
    a_history: List[float] = []
    pi_history: List[float] = []

    pi_prev: Optional[np.ndarray] = None

    # Precompute exponent used in f-update
    alpha = lambda_val / (lambda_val + epsilon)

    for iter_idx in range(max_iters):
        # ---- inner loop init ----
        f = np.ones(n, dtype=float)
        g = np.ones(m, dtype=float)

        prev_dual: Optional[float] = None
        inner_stats: List[Dict[str, float]] = []

        # ---- inner Sinkhorn-like iterations ----
        for inner_step in range(1000):
            f_prev = f.copy()
            g_prev = g.copy()

            # Update f: f = (a / (K g))^alpha
            Kg = K @ g
            Kg_safe = np.maximum(Kg, 1e-300)
            np.power(a / Kg_safe, alpha, out=f)

            # Update g: g = b / (K^T f)
            KTf = K_T @ f
            KTf_safe = np.maximum(KTf, 1e-300)
            np.divide(b, KTf_safe, out=g)

            # monitor diffs
            f_diff = float(np.linalg.norm(f - f_prev))
            g_diff = float(np.linalg.norm(g - g_prev))

            cur_dual = compute_dual_loss(f, K, g, a, b, epsilon, lambda_val)

            if prev_dual is None:
                prev_dual = cur_dual
                continue

            dual_diff = float(cur_dual - prev_dual)
            prev_dual = cur_dual

            if inner_step % 2 == 0:
                inner_stats.append(
                    {
                        "inner_step": float(inner_step),
                        "f_diff": f_diff,
                        "g_diff": g_diff,
                        "dual_diff": dual_diff,
                        "current_Dualloss": float(cur_dual),
                    }
                )

            # Your original convergence criterion:
            if abs(dual_diff) < sinkhorn_tol:
                break

        # Build transport plan pi = diag(f) K diag(g)
        pi = f[:, None] * K * g[None, :]

        # Compute loss components
        transport, neg_entropy, kl, total_loss = compute_loss(pi, C, epsilon, lambda_val, a)

        # Outer update for a: normalize row-sums of pi
        pi_marginal = np.sum(pi, axis=1)
        denom = float(np.sum(pi_marginal))
        if denom <= 0 or not np.isfinite(denom):
            raise FloatingPointError("Invalid normalization encountered in a update (sum(pi_marginal)).")
        a_new = pi_marginal / denom

        a_diff = float(np.linalg.norm(a_new - a))
        a_history.append(a_diff)

        # Track pi change
        if pi_prev is not None:
            pi_history.append(float(np.linalg.norm(pi - pi_prev)))
        pi_prev = pi.copy()

        stats.append(
            {
                "iter": iter_idx + 1,
                "a_diff": a_diff,
                "total_loss": float(total_loss),
                "transport_cost": float(transport),
                "entropy": float(neg_entropy),  # keep your sign convention
                "kl": float(kl),
                "inner_stats": inner_stats,
                "pi_norm": float(np.linalg.norm(pi)),
            }
        )

        # Apply update
        a = a_new

        # Outer stopping criterion (same logic as yours)
        if iter_idx > 0 and abs(stats[-1]["total_loss"] - stats[-2]["total_loss"]) < tol:
            break

        print(f"外层迭代 {iter_idx + 1}/{max_iters}")

    return a, pi, stats, a_history, pi_history
