"""Visualization functions for demand profiling."""

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .profiler import DemandProfile, BenchmarkValidity


def plot_demand_profile(
    profile: DemandProfile,
    title: str = "Benchmark Demand Profile",
    dimensions: Optional[Sequence[str]] = None,
    figsize: tuple = (8, 8),
) -> plt.Figure:
    """Radar chart showing mean demand levels across dimensions.

    Parameters
    ----------
    profile : DemandProfile
    title : str
    dimensions : list of str, optional
        Subset of dimensions to plot. If None, all dimensions.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    if dimensions is not None:
        dims = [d for d in dimensions if d in profile.dimensions]
    else:
        # Filter out dimensions with zero mean
        dims = [d for d in profile.dimensions if profile.mean_demands[d] > 0.01]
    if not dims:
        dims = profile.dimensions

    values = [profile.mean_demands[d] for d in dims]
    n = len(dims)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values = values + [values[0]]
    angles = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    ax.plot(angles, values, "o-", linewidth=2, color="#2196F3")
    ax.fill(angles, values, alpha=0.15, color="#2196F3")

    # Short labels
    labels = [d.replace("_", "\n") for d in dims]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    return fig


def plot_ability_vs_demand(
    ability_profiles: pd.DataFrame,
    demand_profile: DemandProfile,
    model_ids: Optional[Sequence[str]] = None,
    dimensions: Optional[Sequence[str]] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Overlay model ability profiles on benchmark demand profile.

    Parameters
    ----------
    ability_profiles : pd.DataFrame
        (n_models, n_dimensions) ability estimates.
    demand_profile : DemandProfile
    model_ids : list of str, optional
        Subset of models to plot.
    dimensions : list of str, optional
        Subset of dimensions to plot.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    if dimensions is not None:
        dims = [d for d in dimensions if d in demand_profile.dimensions]
    else:
        dims = [d for d in demand_profile.dimensions
                if demand_profile.mean_demands[d] > 0.01]
    if not dims:
        dims = demand_profile.dimensions

    if model_ids is None:
        model_ids = list(ability_profiles.index)

    x = np.arange(len(dims))
    width = 0.8 / (len(model_ids) + 1)

    fig, ax = plt.subplots(figsize=figsize)

    # Demand bars
    demand_vals = [demand_profile.mean_demands[d] for d in dims]
    ax.bar(x - 0.4 + width / 2, demand_vals, width, label="Benchmark Demand",
           color="#E0E0E0", edgecolor="#999")

    # Ability bars
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_ids)))
    for i, model_id in enumerate(model_ids):
        vals = [ability_profiles.loc[model_id, d]
                if d in ability_profiles.columns else 0
                for d in dims]
        ax.bar(x - 0.4 + (i + 1.5) * width, vals, width,
               label=model_id, color=colors[i], alpha=0.8)

    labels = [d.replace("_", "\n") for d in dims]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Level")
    ax.set_title("Model Abilities vs Benchmark Demands")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    return fig


def plot_sensitivity_specificity(
    validity: BenchmarkValidity,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Horizontal bar chart showing sensitivity and specificity scores.

    Parameters
    ----------
    validity : BenchmarkValidity
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    summary = validity.summary()
    target = summary[summary["type"] == "target"].sort_values("score", ascending=True)
    extraneous = summary[summary["type"] == "extraneous"].sort_values("score", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Sensitivity
    if len(target) > 0:
        axes[0].barh(target["dimension"], target["score"], color="#4CAF50", alpha=0.8)
        axes[0].set_xlim(0, 1.1)
        axes[0].axvline(x=0.5, color="red", linestyle="--", alpha=0.5)
        axes[0].set_xlabel("Sensitivity Score")
        axes[0].set_title(f"Sensitivity (Target Dimensions)\nOverall: {validity.overall_sensitivity:.2f}")
    else:
        axes[0].text(0.5, 0.5, "No target dimensions", ha="center", va="center")

    # Specificity
    if len(extraneous) > 0:
        colors = ["#F44336" if s < 0.5 else "#FF9800" if s < 0.75 else "#4CAF50"
                  for s in extraneous["score"]]
        axes[1].barh(extraneous["dimension"], extraneous["score"], color=colors, alpha=0.8)
        axes[1].set_xlim(0, 1.1)
        axes[1].axvline(x=0.75, color="red", linestyle="--", alpha=0.5)
        axes[1].set_xlabel("Specificity Score")
        axes[1].set_title(f"Specificity (Extraneous Dimensions)\nOverall: {validity.overall_specificity:.2f}")
    else:
        axes[1].text(0.5, 0.5, "No extraneous dimensions", ha="center", va="center")

    fig.suptitle(f"Construct Validity: {validity.benchmark_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()

    return fig
