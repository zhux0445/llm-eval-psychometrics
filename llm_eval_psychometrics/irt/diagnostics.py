"""Item diagnostics: classical statistics, item-fit, and diagnostic summaries.

Provides functions for evaluating individual item quality beyond the
basic IRT parameters, including classical test theory (CTT) statistics
and item-fit indices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from llm_eval_psychometrics.irt.models import IRTResults


def classical_item_stats(response_matrix: np.ndarray, item_ids: list[str] | None = None) -> pd.DataFrame:
    """Compute classical test theory statistics for each item.

    These are model-free descriptive statistics that complement IRT analysis:
    - p_value: proportion correct (item easiness)
    - point_biserial: correlation between item score and total score
      (classical measure of discrimination)
    - variance: score variance (maximum at p=0.5)

    Args:
        response_matrix: Binary matrix (n_models, n_items), 1=correct, 0=incorrect.
        item_ids: Optional item labels. Defaults to ['item_0', ...].

    Returns:
        DataFrame with columns ['p_value', 'point_biserial', 'variance'].
    """
    response_matrix = np.asarray(response_matrix, dtype=float)
    n_models, n_items = response_matrix.shape

    if item_ids is None:
        item_ids = [f"item_{i}" for i in range(n_items)]

    # p-value: proportion of models answering correctly
    p_values = np.nanmean(response_matrix, axis=0)

    # Total score per model (excluding the item being evaluated for point-biserial)
    total_scores = np.nansum(response_matrix, axis=1)

    # Point-biserial correlation: correlation between item score and total-minus-item score
    point_biserials = np.zeros(n_items)
    for j in range(n_items):
        item_col = response_matrix[:, j]
        rest_score = total_scores - item_col
        valid = ~np.isnan(item_col)
        if valid.sum() < 3 or np.std(item_col[valid]) == 0:
            point_biserials[j] = np.nan
        else:
            point_biserials[j] = np.corrcoef(item_col[valid], rest_score[valid])[0, 1]

    variances = np.nanvar(response_matrix, axis=0)

    return pd.DataFrame(
        {
            "p_value": p_values,
            "point_biserial": point_biserials,
            "variance": variances,
        },
        index=item_ids,
    )


def item_fit_residuals(results: IRTResults) -> pd.DataFrame:
    """Compute standardized residuals between observed and expected responses.

    For each item, compares the observed proportion correct in each ability
    group to the IRT-predicted probability. Large residuals suggest model
    misfit for that item.

    Uses 5 equal-frequency ability bins. The RMSR (root mean squared residual)
    summarizes fit across bins.

    Args:
        results: Fitted IRTResults object.

    Returns:
        DataFrame with columns ['item_id', 'rmsr'] (root mean squared residual).
        Lower RMSR indicates better fit; values > 0.1 suggest possible misfit.
    """
    response_matrix = results._response_matrix
    theta = results.model_ability.values
    a = results.item_params["discrimination"].values
    b = results.item_params["difficulty"].values
    c = (
        results.item_params["guessing"].values
        if "guessing" in results.item_params.columns
        else np.zeros_like(a)
    )

    n_models, n_items = response_matrix.shape

    # Sort models by ability and create bins
    n_bins = min(5, n_models)
    sorted_idx = np.argsort(theta)
    bins = np.array_split(sorted_idx, n_bins)

    rmsrs = []
    for j in range(n_items):
        sq_residuals = []
        for bin_idx in bins:
            if len(bin_idx) == 0:
                continue
            # Observed proportion correct in this bin
            obs_p = np.mean(response_matrix[bin_idx, j])
            # Expected probability at mean theta of this bin
            mean_theta = np.mean(theta[bin_idx])
            exp_p = c[j] + (1 - c[j]) / (1 + np.exp(-a[j] * (mean_theta - b[j])))
            sq_residuals.append((obs_p - exp_p) ** 2)
        rmsrs.append(np.sqrt(np.mean(sq_residuals)))

    return pd.DataFrame(
        {"rmsr": rmsrs},
        index=results.item_params.index,
    )


def diagnostic_report(results: IRTResults, response_matrix: np.ndarray | None = None) -> pd.DataFrame:
    """Generate a comprehensive item diagnostic report.

    Combines IRT parameters, classical statistics, and fit indices into
    a single DataFrame with a quality flag for each item.

    Quality classification:
    - 'good': discrimination >= 0.5, |difficulty| <= 2.5, RMSR <= 0.1
    - 'acceptable': discrimination >= 0.3, |difficulty| <= 3.0
    - 'poor': everything else

    Args:
        results: Fitted IRTResults object.
        response_matrix: Optional binary matrix. If None, uses the one
            stored in results.

    Returns:
        DataFrame with IRT params, CTT stats, RMSR, and quality classification.
    """
    if response_matrix is None:
        response_matrix = results._response_matrix

    # IRT parameters
    report = results.item_params.copy()

    # Classical stats
    ctt = classical_item_stats(response_matrix, item_ids=list(results.item_params.index))
    report = report.join(ctt)

    # Fit residuals
    fit = item_fit_residuals(results)
    report = report.join(fit)

    # Quality classification
    a = report["discrimination"]
    b_abs = report["difficulty"].abs()
    rmsr = report["rmsr"]

    conditions = [
        (a >= 0.5) & (b_abs <= 2.5) & (rmsr <= 0.1),
        (a >= 0.3) & (b_abs <= 3.0),
    ]
    choices = ["good", "acceptable"]
    report["quality"] = np.select(conditions, choices, default="poor")

    return report
