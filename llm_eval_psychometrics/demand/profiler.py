"""
Multidimensional demand profiling for LLM benchmarks.

Inspired by Zhou et al. (2026) "General scales unlock AI evaluation with
explanatory and predictive power" (Nature, Vol 652).

This module provides tools to:
- Annotate benchmark items across multiple cognitive demand dimensions
- Generate demand profiles quantifying what a benchmark truly measures
- Assess benchmark sensitivity and specificity (construct validity)
- Estimate multidimensional ability profiles for LLMs
- Predict instance-level performance from demand-ability gaps
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr


# Default dimension catalogue based on Zhou et al. (2026), Table S5.
# Users can supply their own dimensions.
DEFAULT_DIMENSIONS = [
    # Elemental capabilities
    "verbal_comprehension",
    "verbal_expression",
    "logical_reasoning",
    "quantitative_reasoning",
    "inductive_reasoning",
    "metacognition_reflection",
    "metacognition_theory_of_mind",
    "metacognition_uncertainty",
    "mind_modelling_social",
    "creativity_elaboration",
    "creativity_exploration",
    # Knowledge
    "knowledge_formal_sciences",
    "knowledge_natural_sciences",
    "knowledge_applied_sciences",
    "knowledge_social_cultural",
    "knowledge_common_sense",
    # Extraneous
    "atypicality",
    "volume",
]


@dataclass
class DemandProfile:
    """Result of demand profiling for a benchmark or subset of items."""

    demand_matrix: pd.DataFrame
    """(n_items, n_dimensions) matrix of demand levels."""

    dimensions: list[str]
    """Names of the demand dimensions."""

    # Per-dimension summary statistics
    mean_demands: pd.Series = field(init=False)
    std_demands: pd.Series = field(init=False)
    median_demands: pd.Series = field(init=False)
    level_distributions: dict[str, pd.Series] = field(init=False)

    def __post_init__(self):
        self.mean_demands = self.demand_matrix.mean()
        self.std_demands = self.demand_matrix.std()
        self.median_demands = self.demand_matrix.median()
        self.level_distributions = {
            dim: self.demand_matrix[dim].value_counts().sort_index()
            for dim in self.dimensions
        }

    def summary(self) -> pd.DataFrame:
        """Return a summary table of demand statistics per dimension."""
        return pd.DataFrame({
            "mean": self.mean_demands,
            "std": self.std_demands,
            "median": self.median_demands,
            "min": self.demand_matrix.min(),
            "max": self.demand_matrix.max(),
            "nonzero_frac": (self.demand_matrix > 0).mean(),
        })


@dataclass
class BenchmarkValidity:
    """Sensitivity and specificity analysis for a benchmark."""

    benchmark_name: str
    target_dimensions: list[str]
    sensitivity: pd.DataFrame
    """Per target dimension: whether the benchmark covers a range of levels."""
    specificity: pd.DataFrame
    """Per non-target dimension: how much extraneous demand is present."""
    overall_sensitivity: float
    overall_specificity: float

    def summary(self) -> pd.DataFrame:
        rows = []
        for _, row in self.sensitivity.iterrows():
            rows.append({
                "dimension": row["dimension"],
                "type": "target",
                "score": row["sensitivity"],
                "detail": row["detail"],
            })
        for _, row in self.specificity.iterrows():
            rows.append({
                "dimension": row["dimension"],
                "type": "extraneous",
                "score": row["specificity"],
                "detail": row["detail"],
            })
        return pd.DataFrame(rows)


class DemandProfiler:
    """Analyze multidimensional demand profiles of benchmark items.

    Parameters
    ----------
    demand_matrix : pd.DataFrame
        DataFrame of shape (n_items, n_dimensions) with demand level
        annotations. Index should be item IDs, columns should be
        dimension names. Values are numeric demand levels (e.g. 0-5).
    response_matrix : np.ndarray or pd.DataFrame, optional
        Binary response matrix of shape (n_models, n_items) for
        ability estimation and performance prediction.
    model_ids : list of str, optional
        Names of models (rows of response_matrix).
    """

    def __init__(
        self,
        demand_matrix: pd.DataFrame,
        response_matrix: Optional[np.ndarray] = None,
        model_ids: Optional[Sequence[str]] = None,
    ):
        if not isinstance(demand_matrix, pd.DataFrame):
            raise TypeError("demand_matrix must be a pandas DataFrame")
        if demand_matrix.empty:
            raise ValueError("demand_matrix must not be empty")

        self.demand_matrix = demand_matrix.copy()
        self.dimensions = list(demand_matrix.columns)
        self.n_items = len(demand_matrix)

        if response_matrix is not None:
            if isinstance(response_matrix, pd.DataFrame):
                response_matrix = response_matrix.values
            response_matrix = np.asarray(response_matrix, dtype=float)
            if response_matrix.ndim != 2:
                raise ValueError("response_matrix must be 2D")
            if response_matrix.shape[1] != self.n_items:
                raise ValueError(
                    f"response_matrix has {response_matrix.shape[1]} items "
                    f"but demand_matrix has {self.n_items} items"
                )
            self.response_matrix = response_matrix
            self.n_models = response_matrix.shape[0]
            if model_ids is not None:
                if len(model_ids) != self.n_models:
                    raise ValueError("model_ids length must match n_models")
                self.model_ids = list(model_ids)
            else:
                self.model_ids = [f"model_{i}" for i in range(self.n_models)]
        else:
            self.response_matrix = None
            self.n_models = 0
            self.model_ids = []

    def demand_profile(self) -> DemandProfile:
        """Compute the demand profile of the benchmark."""
        return DemandProfile(
            demand_matrix=self.demand_matrix.copy(),
            dimensions=self.dimensions,
        )

    def benchmark_validity(
        self,
        benchmark_name: str,
        target_dimensions: Sequence[str],
        min_levels: int = 3,
        max_extraneous_mean: float = 0.5,
    ) -> BenchmarkValidity:
        """Assess benchmark sensitivity and specificity.

        Parameters
        ----------
        benchmark_name : str
            Name of the benchmark being analyzed.
        target_dimensions : list of str
            Dimensions the benchmark claims to measure.
        min_levels : int
            Minimum number of distinct demand levels a target dimension
            should cover for good sensitivity (default 3).
        max_extraneous_mean : float
            Maximum acceptable mean demand for non-target dimensions
            (default 0.5). Higher means worse specificity.

        Returns
        -------
        BenchmarkValidity
        """
        target_dims = list(target_dimensions)
        non_target_dims = [d for d in self.dimensions if d not in target_dims]

        # Sensitivity: do target dimensions cover a range of demand levels?
        sens_rows = []
        for dim in target_dims:
            if dim not in self.demand_matrix.columns:
                sens_rows.append({
                    "dimension": dim,
                    "n_levels": 0,
                    "level_range": 0.0,
                    "sensitivity": 0.0,
                    "detail": "dimension not found in demand matrix",
                })
                continue
            values = self.demand_matrix[dim].dropna()
            n_distinct = values.nunique()
            level_range = values.max() - values.min()
            nonzero_frac = (values > 0).mean()
            # Sensitivity score: fraction of items with nonzero demand *
            # bonus for covering multiple levels
            sens = min(1.0, nonzero_frac * (n_distinct / max(min_levels, 1)))
            sens_rows.append({
                "dimension": dim,
                "n_levels": n_distinct,
                "level_range": level_range,
                "sensitivity": round(sens, 3),
                "detail": f"{n_distinct} levels, range={level_range:.1f}, "
                          f"{nonzero_frac:.1%} nonzero",
            })
        sensitivity_df = pd.DataFrame(sens_rows)

        # Specificity: are non-target dimensions low?
        spec_rows = []
        for dim in non_target_dims:
            if dim not in self.demand_matrix.columns:
                continue
            values = self.demand_matrix[dim].dropna()
            mean_demand = values.mean()
            nonzero_frac = (values > 0).mean()
            # Specificity: 1 - normalized extraneous demand
            spec = max(0.0, 1.0 - mean_demand / max(max_extraneous_mean * 2, 0.01))
            spec_rows.append({
                "dimension": dim,
                "mean_demand": round(mean_demand, 3),
                "nonzero_frac": round(nonzero_frac, 3),
                "specificity": round(spec, 3),
                "detail": f"mean={mean_demand:.2f}, {nonzero_frac:.1%} nonzero",
            })
        specificity_df = pd.DataFrame(spec_rows)

        overall_sens = sensitivity_df["sensitivity"].mean() if len(sensitivity_df) else 0.0
        overall_spec = specificity_df["specificity"].mean() if len(specificity_df) else 1.0

        return BenchmarkValidity(
            benchmark_name=benchmark_name,
            target_dimensions=target_dims,
            sensitivity=sensitivity_df,
            specificity=specificity_df,
            overall_sensitivity=round(overall_sens, 3),
            overall_specificity=round(overall_spec, 3),
        )

    def ability_profile(self) -> pd.DataFrame:
        """Estimate per-dimension ability for each model.

        For each dimension, fits a logistic curve of success probability
        vs demand level, and estimates ability as the demand level at
        which the model has 50% success probability (analogous to IRT
        difficulty/ability matching).

        Returns
        -------
        pd.DataFrame
            (n_models, n_dimensions) ability estimates.
        """
        if self.response_matrix is None:
            raise ValueError("response_matrix required for ability estimation")

        abilities = {}
        for model_idx, model_id in enumerate(self.model_ids):
            model_abilities = {}
            for dim in self.dimensions:
                demand_vals = self.demand_matrix[dim].values
                responses = self.response_matrix[model_idx]

                # Group by demand level bins
                unique_levels = np.sort(np.unique(demand_vals[~np.isnan(demand_vals)]))
                if len(unique_levels) < 2:
                    model_abilities[dim] = np.nan
                    continue

                level_success = []
                level_vals = []
                for lvl in unique_levels:
                    mask = demand_vals == lvl
                    valid = ~np.isnan(responses[mask])
                    if valid.sum() < 1:
                        continue
                    level_success.append(np.nanmean(responses[mask]))
                    level_vals.append(lvl)

                if len(level_vals) < 2:
                    model_abilities[dim] = np.nan
                    continue

                level_vals = np.array(level_vals)
                level_success = np.array(level_success)

                # Fit logistic: P(success) = 1 / (1 + exp(slope * (x - threshold)))
                try:
                    def logistic(x, threshold, slope):
                        return 1.0 / (1.0 + np.exp(slope * (x - threshold)))

                    popt, _ = curve_fit(
                        logistic, level_vals, level_success,
                        p0=[np.median(level_vals), 1.0],
                        bounds=(
                            [level_vals.min() - 2, 0.01],
                            [level_vals.max() + 2, 10.0],
                        ),
                        maxfev=2000,
                    )
                    model_abilities[dim] = round(popt[0], 3)  # threshold = ability
                except (RuntimeError, ValueError):
                    model_abilities[dim] = np.nan

            abilities[model_id] = model_abilities

        return pd.DataFrame(abilities).T

    def predict_performance(
        self,
        ability_profiles: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Predict instance-level performance from demand-ability gaps.

        For each model and item, predicts success probability based on
        how much the model's ability in each dimension exceeds the item's
        demand level. Uses a simple logistic aggregation across dimensions.

        Parameters
        ----------
        ability_profiles : pd.DataFrame, optional
            (n_models, n_dimensions) ability matrix. If None, computed
            from response_matrix via ability_profile().

        Returns
        -------
        pd.DataFrame
            (n_models, n_items) predicted success probabilities.
        """
        if ability_profiles is None:
            ability_profiles = self.ability_profile()

        predictions = {}
        for model_id in ability_profiles.index:
            model_pred = []
            for item_idx in range(self.n_items):
                # Sum of (ability - demand) across dimensions
                gap_sum = 0.0
                n_valid = 0
                for dim in self.dimensions:
                    ability = ability_profiles.loc[model_id, dim]
                    demand = self.demand_matrix.iloc[item_idx][dim]
                    if np.isnan(ability) or np.isnan(demand):
                        continue
                    gap_sum += ability - demand
                    n_valid += 1
                if n_valid == 0:
                    model_pred.append(0.5)
                else:
                    avg_gap = gap_sum / n_valid
                    prob = 1.0 / (1.0 + np.exp(-avg_gap))
                    model_pred.append(round(prob, 4))
            predictions[model_id] = model_pred

        return pd.DataFrame(
            predictions,
            index=self.demand_matrix.index,
        ).T

    def dimension_correlations(self) -> pd.DataFrame:
        """Compute Spearman correlations between demand dimensions.

        Low correlations suggest the dimensions capture distinct constructs.
        """
        corr_matrix = self.demand_matrix.corr(method="spearman")
        return corr_matrix

    def characteristic_curves(self, n_bins: int = 6) -> dict[str, pd.DataFrame]:
        """Compute per-dimension characteristic curves for each model.

        Similar to ICC in IRT but across demand levels for each dimension.

        Returns
        -------
        dict mapping dimension name -> DataFrame with columns for each
        model showing success probability at each demand level bin.
        """
        if self.response_matrix is None:
            raise ValueError("response_matrix required for characteristic curves")

        results = {}
        for dim in self.dimensions:
            demand_vals = self.demand_matrix[dim].values
            valid_mask = ~np.isnan(demand_vals)
            if valid_mask.sum() < 2:
                continue

            # Create bins
            bins = np.linspace(
                np.nanmin(demand_vals), np.nanmax(demand_vals), n_bins + 1
            )
            bin_centers = (bins[:-1] + bins[1:]) / 2

            curves = {"demand_level": bin_centers}
            for model_idx, model_id in enumerate(self.model_ids):
                responses = self.response_matrix[model_idx]
                bin_means = []
                for i in range(len(bins) - 1):
                    if i < len(bins) - 2:
                        mask = valid_mask & (demand_vals >= bins[i]) & (demand_vals < bins[i + 1])
                    else:
                        mask = valid_mask & (demand_vals >= bins[i]) & (demand_vals <= bins[i + 1])
                    valid_resp = responses[mask]
                    valid_resp = valid_resp[~np.isnan(valid_resp)]
                    if len(valid_resp) > 0:
                        bin_means.append(np.mean(valid_resp))
                    else:
                        bin_means.append(np.nan)
                curves[model_id] = bin_means

            results[dim] = pd.DataFrame(curves)

        return results
