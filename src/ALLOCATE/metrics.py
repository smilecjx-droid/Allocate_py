from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

from pathlib import Path
from typing import Optional, Union, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly
import plotly.graph_objects as go
import plotly.express as px

from anndata import AnnData
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import fisher_exact, spearmanr



# -----------------------------------------------------------------------------
# 1) Temporal growth: compute gamma between two timepoints
# -----------------------------------------------------------------------------

def compute_temporal_growth(
    early_obs: pd.DataFrame,
    late_obs: pd.DataFrame,
    *,
    celltype_col: str,
    subtype_parent_map: Optional[Dict[str, str]] = None,
    return_pvalue: bool = False,
    return_counts: bool = False,
) -> pd.DataFrame:
    """
    Compute temporal growth rate (gamma) of each cell type from early to late timepoint.

    Gamma is defined as:
        gamma = (ratio_late - ratio_early) / ratio_early

    where ratio_* is the fraction of cells in that type at the corresponding timepoint.

    Notes
    -----
    - If ratio_early == 0 and ratio_late > 0, gamma = +inf (newly emerging type).
    - If both ratios are 0, gamma = 0.
    - Late timepoint cell types can be merged using `subtype_parent_map`.

    Parameters
    ----------
    early_obs
        Obs DataFrame of early timepoint (e.g., E10.5). Must contain `celltype_col`.
    late_obs
        Obs DataFrame of late timepoint (e.g., E14.5). Must contain `celltype_col`.
    celltype_col
        Column name indicating cell type annotation.
    subtype_parent_map
        Optional mapping from subtype -> parent type for late timepoint merging.
        Example: {"B_sub": "B"}.
    return_pvalue
        If True, include Fisher exact test p-value.
    return_counts
        If True, include early/late counts.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
            - cell_type
            - gamma
        Optionally:
            - p_value
            - count_early, count_late
            - ratio_early, ratio_late
    """
    if celltype_col not in early_obs.columns:
        raise KeyError(f"`{celltype_col}` not found in early_obs.columns.")
    if celltype_col not in late_obs.columns:
        raise KeyError(f"`{celltype_col}` not found in late_obs.columns.")

    subtype_parent_map = subtype_parent_map or {}

    # --- counts ---
    early_counts = early_obs[celltype_col].value_counts()
    late_raw_counts = late_obs[celltype_col].value_counts()

    # --- merge late subtypes into parent types ---
    late_merged = defaultdict(int)
    for subtype, count in late_raw_counts.items():
        parent = subtype_parent_map.get(subtype, subtype)
        late_merged[parent] += int(count)
    late_counts = pd.Series(late_merged, dtype=int)

    total_early = int(early_counts.sum())
    total_late = int(late_counts.sum())

    # We use early cell types as reference (as in your original)
    reference_types = early_obs[celltype_col].unique()

    results: List[Dict[str, Union[str, float, int]]] = []

    for ct in reference_types:
        ct_early = int(early_counts.get(ct, 0))
        ct_late = int(late_counts.get(ct, 0))

        ratio_early = ct_early / total_early if total_early > 0 else 0.0
        ratio_late = ct_late / total_late if total_late > 0 else 0.0

        # --- gamma ---
        if ratio_early == 0.0:
            gamma = 0.0 if ratio_late == 0.0 else np.inf
        else:
            gamma = (ratio_late - ratio_early) / ratio_early

        record: Dict[str, Union[str, float, int]] = {
            "cell_type": ct,
            "gamma": float(gamma),
        }

        if return_pvalue:
            # Fisher exact test on 2x2 contingency table
            _, p = fisher_exact(
                [
                    [ct_early, total_early - ct_early],
                    [ct_late, total_late - ct_late],
                ]
            )
            record["p_value"] = float(p)

        if return_counts:
            record["count_early"] = ct_early
            record["count_late"] = ct_late
            record["ratio_early"] = float(ratio_early)
            record["ratio_late"] = float(ratio_late)

        results.append(record)

    return pd.DataFrame(results)


# -----------------------------------------------------------------------------
# 2) Mapping from OT transport plan to cell-type transfer matrix
# -----------------------------------------------------------------------------

def compute_type_transfer_matrix(
    pi: np.ndarray,
    t1_obs: pd.DataFrame,
    t2_obs: pd.DataFrame,
    *,
    celltype_col: str,
) -> pd.DataFrame:
    """
    Aggregate a cell-level transport plan into a cell-type transfer matrix.

    Robust implementation using np.add.at (no bincount, no flatten indices).
    """
    import numpy as np
    import pandas as pd

    if celltype_col not in t1_obs.columns:
        raise KeyError(f"`{celltype_col}` not found in t1_obs.")
    if celltype_col not in t2_obs.columns:
        raise KeyError(f"`{celltype_col}` not found in t2_obs.")

    pi = np.asarray(pi, dtype=float)

    # ✅ 用字符串重新构建 categorical，避免历史 categories 导致 -1
    src_cat = pd.Categorical(t1_obs[celltype_col].astype(str))
    tgt_cat = pd.Categorical(t2_obs[celltype_col].astype(str))

    src_types = src_cat.categories
    tgt_types = tgt_cat.categories

    src_codes = src_cat.codes
    tgt_codes = tgt_cat.codes

    K1 = len(src_types)
    K2 = len(tgt_types)

    transfer = np.zeros((K1, K2), dtype=float)

    # ✅ 最稳聚合：不会出现负 index flatten
    np.add.at(transfer, (src_codes[:, None], tgt_codes[None, :]), pi)

    return pd.DataFrame(transfer, index=src_types, columns=tgt_types)


def infer_parent_mapping_from_transfer(
    type_transfer: pd.DataFrame,
) -> Dict[str, str]:
    """
    Infer parent mapping for newly appearing cell types in target timepoint.

    For each target type not present in source types, choose the source type
    with maximum incoming mass as its "parent".

    Parameters
    ----------
    type_transfer
        Cell-type transfer matrix (rows: source types, columns: target types).

    Returns
    -------
    dict
        Mapping {new_target_type: inferred_parent_type}.
    """
    t1_types = set(type_transfer.index)
    t2_types = list(type_transfer.columns)

    new_types = [t for t in t2_types if t not in t1_types]

    parent_mapping: Dict[str, str] = {}
    for new_type in new_types:
        parent_mapping[new_type] = type_transfer[new_type].idxmax()

    return parent_mapping


def get_mapping(
    pi: np.ndarray,
    t1,
    t2,
    celltype_col: str,
) -> Tuple[Dict[str, str], pd.DataFrame]:
    """
    Wrapper compatible with your original signature.

    Parameters
    ----------
    pi
        OT coupling matrix
    t1, t2
        AnnData-like objects with `.obs` and `.obs_names` aligning with pi.
    celltype_col
        Column name in obs storing cell type.

    Returns
    -------
    parent_mapping
        Inferred parent mapping for new cell types.
    type_transfer
        Cell-type transfer matrix.
    """
    # Ensure pi is numpy array
    pi = np.asarray(pi, dtype=float)

    # Compute transfer matrix
    type_transfer = compute_type_transfer_matrix(
        pi,
        t1.obs,
        t2.obs,
        celltype_col=celltype_col,
    )

    # Infer new cell type parent mapping
    parent_mapping = infer_parent_mapping_from_transfer(type_transfer)

    return parent_mapping, type_transfer


# -----------------------------------------------------------------------------
# 3) Spearman correlation plot between a_means and gamma
# -----------------------------------------------------------------------------

def spearman_scatter_plot(
    a_means_df: pd.DataFrame,
    gamma_df: pd.DataFrame,
    *,
    x_col: str = "a_means",
    y_col: str = "gamma",
    merge_key: str = "cell_type",
    hue_col: Optional[str] = None,
    title: str = "Spearman correlation",
    figsize: Tuple[float, float] = (7.0, 5.0),
    point_size: int = 120,
    add_regression: bool = True,
    palette: str = "tab20",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> Tuple[float, float, plt.Figure]:
    """
    Plot scatter of a_means vs gamma and compute Spearman correlation.

    Parameters
    ----------
    a_means_df
        DataFrame containing at least [merge_key, x_col].
    gamma_df
        DataFrame containing at least [merge_key, y_col].
    x_col
        Column name for x-axis values (e.g. a_means).
    y_col
        Column name for y-axis values (e.g. gamma).
    merge_key
        Column used to merge (default "cell_type").
    hue_col
        Optional column used for coloring points. If None, no hue.
    title
        Plot title.
    figsize
        Figure size.
    point_size
        Scatter point size.
    add_regression
        Whether to add a linear regression trend line.
    palette
        Seaborn palette name.
    ax
        Existing axis to plot on.
    show
        Whether to display the plot.

    Returns
    -------
    spearman_r
        Spearman correlation coefficient.
    spearman_p
        Spearman p-value.
    fig
        Matplotlib figure object.
    """
    # Merge & drop missing
    merged = pd.merge(a_means_df, gamma_df, on=merge_key, how="inner")
    merged = merged.dropna(subset=[x_col, y_col])

    if merged.empty:
        raise ValueError("No overlapping rows after merge. Check merge_key and input DataFrames.")

    # Spearman correlation
    spearman_r, spearman_p = spearmanr(merged[x_col], merged[y_col])

    # Setup axis/figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sns.scatterplot(
        data=merged,
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette=palette if hue_col is not None else None,
        s=point_size,
        edgecolor="w",
        linewidth=1,
        ax=ax,
    )

    if add_regression:
        sns.regplot(
            data=merged,
            x=x_col,
            y=y_col,
            scatter=False,
            color="blue",
            line_kws={"lw": 2, "alpha": 0.7},
            ax=ax,
        )

    stats_text = f"Spearman ρ = {spearman_r:.3f}\nP-value = {spearman_p:.3e}"
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("a mean", fontsize=12)
    ax.set_ylabel("Gamma value", fontsize=12)

    if hue_col is not None:
        ax.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        ax.legend_.remove() if ax.legend_ is not None else None

    fig.tight_layout()

    if show:
        plt.show()

    return float(spearman_r), float(spearman_p), fig


# -----------------------------------------------------------------------------
# 4) The heatmap and accuracy of transition matrix
# -----------------------------------------------------------------------------



def aggregate_pi_by_celltype(
    t1: AnnData,
    t2: AnnData,
    pi: np.ndarray,
    *,
    celltype_key: str = "cell_type",
    normalize: Optional[str] = None,
) -> pd.DataFrame:
    """
    Aggregate a cell-level transport plan into a cell-type transfer matrix.

    Parameters
    ----------
    t1, t2
        AnnData objects for source and target timepoints.
        The order of `t1.obs_names` must match the row order of `pi`,
        and `t2.obs_names` must match the column order of `pi`.
    pi
        Transport plan of shape (n_source_cells, n_target_cells).
    celltype_key
        Column in t1.obs / t2.obs that stores cell type labels.
    normalize
        Optional normalization:
        - None: keep raw summed weights
        - "row": each source cell-type row sums to 1
        - "col": each target cell-type column sums to 1
        - "all": whole matrix sums to 1

    Returns
    -------
    pd.DataFrame
        Cell-type transfer matrix (source_types x target_types).
    """
    if celltype_key not in t1.obs:
        raise KeyError(f"`{celltype_key}` not found in t1.obs.")
    if celltype_key not in t2.obs:
        raise KeyError(f"`{celltype_key}` not found in t2.obs.")

    pi = np.asarray(pi, dtype=float)
    if pi.shape != (t1.n_obs, t2.n_obs):
        raise ValueError(
            f"`pi` shape {pi.shape} does not match (t1.n_obs, t2.n_obs)=({t1.n_obs}, {t2.n_obs})."
        )

    # Encode cell types into integer codes
    src_cat = pd.Categorical(t1.obs[celltype_key])
    tgt_cat = pd.Categorical(t2.obs[celltype_key])

    src_types = src_cat.categories
    tgt_types = tgt_cat.categories

    src_codes = src_cat.codes  # shape (n_source,)
    tgt_codes = tgt_cat.codes  # shape (n_target,)

    K1 = len(src_types)
    K2 = len(tgt_types)

    # Aggregate using np.add.at (fast and memory efficient)
    transfer = np.zeros((K1, K2), dtype=float)
    # For each cell pair (i, j): add pi[i, j] to transfer[src_codes[i], tgt_codes[j]]
    np.add.at(transfer, (src_codes[:, None], tgt_codes[None, :]), pi)

    transfer_df = pd.DataFrame(transfer, index=src_types, columns=tgt_types)

    # Optional normalization
    if normalize is not None:
        normalize = normalize.lower()
        if normalize == "row":
            transfer_df = transfer_df.div(transfer_df.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        elif normalize == "col":
            transfer_df = transfer_df.div(transfer_df.sum(axis=0).replace(0, np.nan), axis=1).fillna(0)
        elif normalize == "all":
            s = transfer_df.values.sum()
            transfer_df = transfer_df / s if s > 0 else transfer_df
        else:
            raise ValueError("normalize must be one of {None, 'row', 'col', 'all'}.")

    return transfer_df



def pi_process(t1, t2, pi, key: str = "cell_type") -> pd.DataFrame:
    """
    Aggregate a transport plan matrix `pi` by source/target cell types.

    Parameters
    ----------
    t1 : AnnData
        Source AnnData object. Must contain `t1.obs[key]`.
    t2 : AnnData
        Target AnnData object. Must contain `t2.obs[key]`.
    pi : array-like, shape (n_source, n_target)
        Transport plan / coupling matrix between source cells and target cells.
    key : str, default "cell_type"
        Column name in `.obs` used for grouping cell types.

    Returns
    -------
    flow_df : pd.DataFrame
        A DataFrame with columns: ["source", "target", "weight"].
        Each row corresponds to the total transport weight aggregated
        between source and target cell types.
    """
    # ---------- Input checks ----------
    if key not in t1.obs.columns:
        raise KeyError(f"Key '{key}' not found in t1.obs.columns.")
    if key not in t2.obs.columns:
        raise KeyError(f"Key '{key}' not found in t2.obs.columns.")

    pi = np.asarray(pi)
    if pi.ndim != 2:
        raise ValueError(f"`pi` must be a 2D matrix. Got pi.ndim = {pi.ndim}.")

    n_source, n_target = pi.shape
    if n_source != t1.n_obs:
        raise ValueError(
            f"Mismatch: pi.shape[0] = {n_source}, but t1.n_obs = {t1.n_obs}."
        )
    if n_target != t2.n_obs:
        raise ValueError(
            f"Mismatch: pi.shape[1] = {n_target}, but t2.n_obs = {t2.n_obs}."
        )

    # ---------- Expand labels to match flattened pi ----------
    source_labels = t1.obs[key].to_numpy()
    target_labels = t2.obs[key].to_numpy()

    source_expanded = np.repeat(source_labels, n_target)
    target_expanded = np.tile(target_labels, n_source)
    pi_flat = pi.ravel()

    # ---------- Build DataFrame and aggregate ----------
    pi_df = pd.DataFrame(
        {"source": source_expanded, "target": target_expanded, "weight": pi_flat}
    )

    flow_df = (
        pi_df.groupby(["source", "target"], as_index=False)["weight"]
        .sum()
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )

    return flow_df



def compute_mapping_accuracy(
    flow_df: pd.DataFrame,
    *,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: str = "weight",
    return_details: bool = False,
) -> float | Tuple[float, pd.DataFrame, float]:
    """
    Compute mapping accuracy based on diagonal mass in a flow DataFrame.

    The accuracy is defined as:
        accuracy = sum(weight where source == target) / sum(all weights)

    Parameters
    ----------
    flow_df
        DataFrame containing mapping flow with columns [source, target, weight].
        Each row represents the aggregated mass transferred from `source` to `target`.
    source_col
        Column name for source group label.
    target_col
        Column name for target group label.
    weight_col
        Column name for flow weight.
    return_details
        If True, also return:
            - diagonal_df (flow entries where source == target)
            - total_weight

    Returns
    -------
    float
        Accuracy value in [0, 1] if total mass > 0.
    (optional) tuple
        (accuracy, diagonal_df, total_weight)
    """
    required_cols = {source_col, target_col, weight_col}
    missing = required_cols - set(flow_df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    total_weight = float(flow_df[weight_col].sum())
    if total_weight == 0.0:
        accuracy = 0.0
        diagonal_df = flow_df.iloc[0:0].copy()
        return (accuracy, diagonal_df, total_weight) if return_details else accuracy

    diagonal_df = flow_df.loc[flow_df[source_col] == flow_df[target_col]]
    accuracy = float(diagonal_df[weight_col].sum())

    return accuracy
