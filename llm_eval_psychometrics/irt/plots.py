"""IRT visualization: Item Information Curves (IIC), Test Information Curves (TIC),
Item Characteristic Curves (ICC), and Wright Maps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from llm_eval_psychometrics.irt.models import IRTResults


def plot_icc(
    results: IRTResults,
    item_ids: list[str] | None = None,
    theta: np.ndarray | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot Item Characteristic Curves (ICC).

    The ICC shows the probability of a correct response as a function of
    ability (theta). Steeper curves indicate higher discrimination.

    Args:
        results: Fitted IRTResults object.
        item_ids: Items to plot. Defaults to all items.
        theta: Ability values for the x-axis. Defaults to linspace(-4, 4, 201).
        ax: Optional matplotlib Axes. Creates a new figure if None.

    Returns:
        The matplotlib Figure containing the plot.
    """
    if theta is None:
        theta = np.linspace(-4, 4, 201)

    params = results.item_params
    if item_ids is None:
        item_ids = list(params.index)

    a = params.loc[item_ids, "discrimination"].values
    b = params.loc[item_ids, "difficulty"].values
    c = (
        params.loc[item_ids, "guessing"].values
        if "guessing" in params.columns
        else np.zeros_like(a)
    )

    fig, ax = _get_fig_ax(ax)

    # P(θ) = c + (1 - c) / (1 + exp(-a(θ - b)))
    theta_2d = theta[:, np.newaxis]
    p = c + (1 - c) / (1 + np.exp(-a * (theta_2d - b)))

    for i, item_id in enumerate(item_ids):
        ax.plot(theta, p[:, i], label=item_id)

    ax.set_xlabel("Ability (θ)")
    ax.set_ylabel("P(correct)")
    ax.set_title("Item Characteristic Curves")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize="small", loc="best", ncol=max(1, len(item_ids) // 10))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_iic(
    results: IRTResults,
    item_ids: list[str] | None = None,
    theta: np.ndarray | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot Item Information Curves (IIC).

    Item information quantifies measurement precision at each ability level.
    Peaks indicate where the item is most useful for distinguishing ability.

    Args:
        results: Fitted IRTResults object.
        item_ids: Items to plot. Defaults to all items.
        theta: Ability values for the x-axis. Defaults to linspace(-4, 4, 201).
        ax: Optional matplotlib Axes. Creates a new figure if None.

    Returns:
        The matplotlib Figure containing the plot.
    """
    if theta is None:
        theta = np.linspace(-4, 4, 201)

    if item_ids is None:
        item_ids = list(results.item_params.index)

    info_df = results.item_information(theta)

    fig, ax = _get_fig_ax(ax)

    for item_id in item_ids:
        ax.plot(theta, info_df[item_id].values, label=item_id)

    ax.set_xlabel("Ability (θ)")
    ax.set_ylabel("Information I(θ)")
    ax.set_title("Item Information Curves")
    ax.legend(fontsize="small", loc="best", ncol=max(1, len(item_ids) // 10))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_tic(
    results: IRTResults,
    theta: np.ndarray | None = None,
    show_se: bool = True,
    ax: Axes | None = None,
) -> Figure:
    """Plot the Test Information Curve (TIC) and optional Standard Error.

    Test information is the sum of all item information functions.
    The standard error of ability estimation is SE(θ) = 1 / sqrt(I(θ)).

    Args:
        results: Fitted IRTResults object.
        theta: Ability values for the x-axis. Defaults to linspace(-4, 4, 201).
        show_se: If True, plot standard error on a secondary y-axis.
        ax: Optional matplotlib Axes. Creates a new figure if None.

    Returns:
        The matplotlib Figure containing the plot.
    """
    if theta is None:
        theta = np.linspace(-4, 4, 201)

    tic = results.test_information(theta)

    fig, ax = _get_fig_ax(ax)

    color_info = "steelblue"
    ax.plot(theta, tic.values, color=color_info, linewidth=2, label="Test Information")
    ax.set_xlabel("Ability (θ)")
    ax.set_ylabel("Information I(θ)", color=color_info)
    ax.tick_params(axis="y", labelcolor=color_info)
    ax.set_title("Test Information Curve")
    ax.grid(True, alpha=0.3)

    if show_se:
        # SE(θ) = 1 / sqrt(I(θ)), undefined where I(θ) ≈ 0
        with np.errstate(divide="ignore", invalid="ignore"):
            se = np.where(tic.values > 1e-10, 1.0 / np.sqrt(tic.values), np.nan)

        color_se = "firebrick"
        ax2 = ax.twinx()
        ax2.plot(theta, se, color=color_se, linewidth=1.5, linestyle="--", label="SE(θ)")
        ax2.set_ylabel("Standard Error", color=color_se)
        ax2.tick_params(axis="y", labelcolor=color_se)

        # Combined legend
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    fig.tight_layout()
    return fig


def plot_wright_map(
    results: IRTResults,
    ax: Axes | None = None,
) -> Figure:
    """Plot a Wright Map (item-person map).

    Shows the distribution of model abilities and item difficulties on the
    same logit scale, making it easy to see whether items are well-targeted
    to the ability range of the models being evaluated.

    Args:
        results: Fitted IRTResults object.
        ax: Optional matplotlib Axes. Creates a new figure if None.

    Returns:
        The matplotlib Figure containing the plot.
    """
    fig, ax = _get_fig_ax(ax, figsize=(8, 6))

    abilities = results.model_ability.values
    difficulties = results.item_params["difficulty"].values
    item_labels = list(results.item_params.index)

    # Left side: model ability distribution (histogram)
    y_range = (
        min(abilities.min(), difficulties.min()) - 0.5,
        max(abilities.max(), difficulties.max()) + 0.5,
    )

    ax.hist(
        abilities,
        bins=15,
        orientation="horizontal",
        alpha=0.6,
        color="steelblue",
        label="Model abilities",
        density=True,
    )

    # Right side: item difficulties as points
    x_max = ax.get_xlim()[1]
    jitter = np.random.default_rng(0).uniform(-0.02, 0.02, len(difficulties))
    ax.scatter(
        np.full_like(difficulties, x_max * 1.1) + jitter,
        difficulties,
        marker="d",
        color="firebrick",
        s=40,
        zorder=5,
        label="Item difficulties",
    )

    for i, label in enumerate(item_labels):
        ax.annotate(
            label,
            (x_max * 1.15, difficulties[i]),
            fontsize=6,
            va="center",
        )

    ax.set_ylabel("Logit Scale (θ / b)")
    ax.set_xlabel("Density")
    ax.set_title("Wright Map: Models vs Items")
    ax.legend(loc="upper left", fontsize="small")
    ax.set_ylim(*y_range)
    fig.tight_layout()
    return fig


def _get_fig_ax(
    ax: Axes | None, figsize: tuple[float, float] = (10, 5)
) -> tuple[Figure, Axes]:
    """Get or create a Figure and Axes pair.

    Args:
        ax: Existing Axes or None.
        figsize: Figure size if creating new.

    Returns:
        (Figure, Axes) tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax
