from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Literal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import seaborn as sns
from anndata import AnnData
import pandas as pd
import scanpy as sc

def plot_ALLOCATE_a_distribution(
    adata_or_a: Union[AnnData, np.ndarray],
    *,
    key: str = "ot_a",
    title: str = "OT Marginal Distribution",
    bins: int = 50,
    kde: bool = True,
    log_eps: float = 1e-12,
    show_uniform: bool = True,
    figsize: tuple[float, float] = (4.2, 3.2),
    savepath: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot the distribution of OT marginal weights `a` in log10 scale.

    Parameters
    ----------
    adata_or_a
        Either an AnnData object (read weights from `adata.obs[key]`) or a 1D numpy array `a`.
    key
        Column name in `adata.obs` where marginal weights are stored (only used when input is AnnData).
    title
        Plot title.
    bins
        Number of histogram bins.
    kde
        Whether to overlay KDE curve.
    log_eps
        Small offset to avoid log10(0).
    show_uniform
        If True, draw a vertical line showing the theoretical uniform value log10(1/n).
    figsize
        Figure size used when `ax` is None.
    savepath
        If provided, save the figure to this file (supports .pdf/.png/.jpg etc.).
    dpi
        DPI for saved figure.
    ax
        Existing matplotlib axis to draw on. If None, create a new figure and axis.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the plot.
    """
    # --- Extract a values ---
    if isinstance(adata_or_a, AnnData):
        if key not in adata_or_a.obs:
            raise KeyError(f"`{key}` not found in `adata.obs`. Available keys: {list(adata_or_a.obs.keys())[:10]}...")
        a_values = np.asarray(adata_or_a.obs[key].values, dtype=float)
        n_samples = adata_or_a.n_obs
    else:
        a_values = np.asarray(adata_or_a, dtype=float).ravel()
        n_samples = a_values.shape[0]

    if a_values.ndim != 1:
        raise ValueError(f"`a` must be 1D, got shape {a_values.shape}.")

    # --- Numerical safety: log10 ---
    a_safe = np.maximum(a_values, log_eps)
    a_log10 = np.log10(a_safe)

    # theoretical uniform log10(1/n)
    uniform_log10 = np.log10(1.0 / n_samples)

    # --- Setup axes ---
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    # --- Plot ---
    sns.histplot(a_log10, bins=bins, kde=kde, ax=ax)

    if show_uniform:
        ax.axvline(
            uniform_log10,
            linestyle="--",
            linewidth=1,
            label=f"Uniform theory: {uniform_log10:.2f}",
        )
        ax.legend()

    ax.set_xlabel("log10(a)", fontsize=10)
    ax.set_ylabel("Cell count", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)

    # --- Save ---
    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(savepath, dpi=dpi, bbox_inches="tight")

    # if we created the fig, return ax
    return ax



def plot_ALLOCATE_a_umap(
    adata: AnnData,
    *,
    a: Optional[np.ndarray] = None,
    a_key: str = "ot_a",
    group_key: str = "cell_type",
    group_mean_key: str = "a_mean_by_group",
    umap_key: str = "X_umap",
    title: str = "UMAP colored by mean(a) per group",
    point_size: float = 2.0,
    center: str = "mid",  # options: "mid", "mean", "median", or numeric
    cmap: Optional[LinearSegmentedColormap] = None,
    savepath: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a UMAP embedding colored by group-mean of OT marginal weights `a`.

    This function computes, for each cell:
        a_mean(cell) = mean_{cells in same group}(a)

    and colors the UMAP coordinates by `a_mean`.

    Parameters
    ----------
    adata
        AnnData containing UMAP coordinates in `adata.obsm[umap_key]`.
    a
        Optional 1D array of marginal weights with length = adata.n_obs.
        If provided, it will be written into `adata.obs[a_key]`.
        If None, the function reads values from `adata.obs[a_key]`.
    a_key
        Column name in `adata.obs` storing the marginal weights `a`.
    group_key
        Column in `adata.obs` specifying groups (e.g., cell_type).
    group_mean_key
        Column name to store group mean values in `adata.obs`.
    umap_key
        Key in `adata.obsm` storing UMAP coordinates.
    title
        Plot title.
    point_size
        Scatter marker size.
    center
        How to choose the center for TwoSlopeNorm:
        - "mid": midpoint between min and max
        - "mean": mean of values
        - "median": median of values
        - numeric: explicit float center
    cmap
        Colormap. If None, uses a custom blue→red colormap similar to yours.
    savepath
        If provided, saves the figure to this path.
    dpi
        Save DPI.
    ax
        Existing matplotlib axis. If None, a new figure is created.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the plot.
    """
    # --- Validate UMAP coordinates ---
    if umap_key not in adata.obsm:
        raise KeyError(f"`{umap_key}` not found in `adata.obsm`.")
    umap = np.asarray(adata.obsm[umap_key])
    if umap.shape[1] < 2:
        raise ValueError(f"`adata.obsm['{umap_key}']` must have at least 2 columns.")

    # --- Load/assign a values ---
    if a is not None:
        a = np.asarray(a, dtype=float).ravel()
        if a.shape[0] != adata.n_obs:
            raise ValueError(f"`a` length {a.shape[0]} does not match adata.n_obs={adata.n_obs}.")
        adata.obs[a_key] = a
    else:
        if a_key not in adata.obs:
            raise KeyError(f"`{a_key}` not found in `adata.obs`. Provide `a=` or store it in obs first.")
        a = np.asarray(adata.obs[a_key].values, dtype=float)

    # --- Validate grouping key ---
    if group_key not in adata.obs:
        raise KeyError(f"`{group_key}` not found in `adata.obs`.")

    # --- Compute group mean for each cell ---
    adata.obs[group_mean_key] = adata.obs.groupby(group_key)[a_key].transform("mean")
    values = np.asarray(adata.obs[group_mean_key].values, dtype=float)

    # --- Colormap ---
    if cmap is None:
        cmap = LinearSegmentedColormap.from_list(
            "custom_cmap",
            ["#000080", "#87CEEB", "#FFFACD", "#FFA500", "darkred"],
        )

    # --- Normalization for color ---
    vmin, vmax = float(values.min()), float(values.max())
    if isinstance(center, str):
        center = center.lower()
        if center == "mid":
            vcenter = 0.5 * (vmin + vmax)
        elif center == "mean":
            vcenter = float(values.mean())
        elif center == "median":
            vcenter = float(np.median(values))
        else:
            raise ValueError("center must be 'mid', 'mean', 'median', or a numeric value.")
    else:
        vcenter = float(center)

    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    # --- Setup axis ---
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        created_fig = True

    # --- Scatter plot ---
    sc = ax.scatter(
        umap[:, 0],
        umap[:, 1],
        c=values,
        cmap=cmap,
        norm=norm,
        s=point_size,
        linewidths=0,
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])

    # --- Colorbar ---
    cbar = ax.figure.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label(f"mean({a_key}) per {group_key}", fontsize=10)

    # --- Save ---
    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return ax




def plot_ALLOCATE_a_violin(
    adata: AnnData,
    *,
    a: Optional[np.ndarray] = None,
    a_key: str = "ot_a",
    value_key: str = "logot_a",
    group_key: str = "cell_type",
    sort_by: Literal["mean", "median"] = "mean",
    descending: bool = True,
    log10: bool = True,
    log_eps: float = 1e-12,
    inner: str = "box",
    stripplot: bool = False,
    linewidth: float = 0.2,
    figsize: tuple[float, float] = (4.2, 2.6),
    rotation: int = 55,
    fontsize: int = 6,
    title: str = "OT a distribution by cell type",
    ylabel: str = "log10(ot_a)",
    savepath: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot a violin plot of OT marginal weights `a` grouped by cell type.

    The plot values can be:
        - log10(a) (default)
        - or raw a

    Cell types are optionally sorted by the group mean/median of `a`.

    Parameters
    ----------
    adata
        AnnData object containing cell metadata.
    a
        Optional marginal weights array of shape (n_cells,). If provided, will be stored in `adata.obs[a_key]`.
        If None, function reads from `adata.obs[a_key]`.
    a_key
        Column name to store/read raw OT marginal weights.
    value_key
        Column name used for plotting (e.g., "logot_a").
    group_key
        Column in `adata.obs` containing group labels (e.g., cell type).
    sort_by
        How to rank cell types: "mean" or "median".
    descending
        Whether to sort descending.
    log10
        If True, plot log10(a). Otherwise plot raw a.
    log_eps
        Small epsilon added to a before log10 to avoid log(0).
    inner
        Inner representation in violin plot ("box", "quartile", etc).
    stripplot
        If True, plot individual points (slow for large data).
    linewidth
        Line width for violin/box.
    figsize
        Figure size when ax is None.
    rotation
        Rotation angle for x tick labels.
    fontsize
        Font size for labels.
    title
        Plot title.
    ylabel
        Y axis label.
    savepath
        If provided, save figure to this path.
    dpi
        Save DPI.
    ax
        Matplotlib axis. If None, creates a new figure.
    show
        If True, call plt.show() at end.

    Returns
    -------
    ax
        Matplotlib axis containing the violin plot.
    """
    if group_key not in adata.obs:
        raise KeyError(f"`{group_key}` not found in adata.obs.")

    # --- Load/assign a values ---
    if a is not None:
        a = np.asarray(a, dtype=float).ravel()
        if a.shape[0] != adata.n_obs:
            raise ValueError(f"`a` length {a.shape[0]} does not match adata.n_obs={adata.n_obs}.")
        adata.obs[a_key] = a
    else:
        if a_key not in adata.obs:
            raise KeyError(f"`{a_key}` not found in adata.obs. Provide `a=` or store it first.")
        a = np.asarray(adata.obs[a_key].values, dtype=float)

    # --- Build plotting values ---
    if log10:
        adata.obs[value_key] = np.log10(np.maximum(a, log_eps))
    else:
        adata.obs[value_key] = a

    # --- Sort cell types by mean/median of raw `a_key` ---
    if sort_by == "mean":
        order = adata.obs.groupby(group_key)[a_key].mean().sort_values(ascending=not descending).index.tolist()
    elif sort_by == "median":
        order = adata.obs.groupby(group_key)[a_key].median().sort_values(ascending=not descending).index.tolist()
    else:
        raise ValueError("sort_by must be 'mean' or 'median'.")

    # enforce categorical order (does not overwrite original categories permanently unless stored)
    adata.obs[group_key] = pd.Categorical(adata.obs[group_key], categories=order, ordered=True)

    # --- Setup axis ---
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        created_fig = True

    # --- Plot using scanpy ---
    sc.pl.violin(
        adata,
        keys=value_key,
        groupby=group_key,
        stripplot=stripplot,
        inner=inner,
        linewidth=linewidth,
        ax=ax,
        show=False,
    )

    # --- Format axes ---
    ax.set_title(title, fontsize=fontsize + 1, pad=8)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=fontsize)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=rotation, ha="right", fontsize=fontsize)

    # --- Save ---
    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(savepath, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return ax


def plot_ALLOCATE_transfer_heatmap(
    transfer_df: pd.DataFrame,
    *,
    title: str = "Cell-type Transition Heatmap",
    xlabel: str = "Target Cell Type",
    ylabel: str = "Source Cell Type",
    figsize: tuple[float, float] = (8, 6),
    cmap: Optional[LinearSegmentedColormap] = None,
    annot: bool = False,
    fmt: str = ".2f",
    linewidths: float = 0.0,
    cbar_label: str = "Weight",
    savepath: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot a heatmap for a cell-type transfer matrix.

    Behavior (as requested):
    - No auto-fill of missing types (no union, no fillna).
    - Align common cell types so they appear in the same order on both axes.
    - Types appearing only on one axis remain at the end in their original order.
    - No diagonal sorting, no normalization, no log transforms.

    Parameters
    ----------
    transfer_df
        DataFrame where rows = source types and columns = target types.
    """
    df = transfer_df.copy()

    row_labels = list(df.index)
    col_labels = list(df.columns)

    col_set = set(col_labels)
    row_set = set(row_labels)

    # Common types follow row order
    common = [ct for ct in row_labels if ct in col_set]

    if len(common) == 0:
        raise ValueError("No overlapping cell types between rows and columns.")

    # Keep row-only / col-only types at the end (original order preserved)
    row_only = [ct for ct in row_labels if ct not in col_set]
    col_only = [ct for ct in col_labels if ct not in row_set]

    row_order = common + row_only
    col_order = common + col_only

    df = df.loc[row_order, col_order]

    # Default colormap
    if cmap is None:
        colors = ["#F8F8F8", "#90EE90", "#2E8B57", "#006400"]
        cmap = LinearSegmentedColormap.from_list("white_to_darkgreen", colors)

    # Setup axis
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sns.heatmap(
        df,
        xticklabels=df.columns,
        yticklabels=df.index,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        linewidths=linewidths,
        cbar_kws={"label": cbar_label},
        ax=ax,
    )

    ax.set_title(title, fontsize=14, pad=16)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    fig.tight_layout()

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return ax
