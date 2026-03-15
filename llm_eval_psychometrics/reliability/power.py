"""Sample size planning and bootstrap confidence intervals.

Provides tools for determining how many benchmark items or evaluation
samples are needed for reliable measurement, and for computing
bootstrap confidence intervals on evaluation scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PowerResult:
    """Result of a power/sample-size calculation.

    Attributes:
        metric: What was calculated.
        value: The computed value (e.g., minimum sample size).
        details: Description of parameters and assumptions.
    """

    metric: str
    value: float
    details: str

    def __repr__(self) -> str:
        return f"PowerResult({self.metric} = {self.value}, {self.details})"


@dataclass
class BootstrapCIResult:
    """Result of bootstrap confidence interval estimation.

    Attributes:
        estimate: Point estimate (e.g., mean).
        ci_lower: Lower bound of confidence interval.
        ci_upper: Upper bound of confidence interval.
        ci_level: Confidence level (e.g., 0.95).
        se: Bootstrap standard error.
        n_bootstrap: Number of bootstrap resamples used.
    """

    estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    se: float
    n_bootstrap: int

    def __repr__(self) -> str:
        return (
            f"BootstrapCIResult(estimate={self.estimate:.4f}, "
            f"{self.ci_level:.0%} CI [{self.ci_lower:.4f}, {self.ci_upper:.4f}], "
            f"SE={self.se:.4f})"
        )


class PowerCalculator:
    """Sample size planning and statistical power for LLM evaluations.

    Helps answer questions like:
    - How many benchmark items do I need for precise ability estimation?
    - How many evaluation samples are needed to detect a score difference?
    - What's the confidence interval on my evaluation results?

    Example:
        >>> power = PowerCalculator()
        >>> power.min_items_for_precision(target_se=0.1)
        >>> power.min_samples_for_difference(effect_size=0.5)
        >>> power.bootstrap_ci(scores, n_bootstrap=2000)
    """

    def min_items_for_precision(
        self,
        target_se: float = 0.1,
        ability_range: tuple[float, float] = (-3, 3),
        mean_discrimination: float = 1.0,
        mean_difficulty: float = 0.0,
    ) -> PowerResult:
        """Estimate minimum number of IRT items for a target standard error.

        Uses the IRT information function to determine how many items are
        needed to achieve a desired precision of ability estimation.

        For a 2PL model, each item provides information:
            I(θ) = a² * P(θ) * (1 - P(θ))
        The SE at ability θ is: SE(θ) = 1 / sqrt(n * mean_I(θ))

        Args:
            target_se: Desired standard error of ability estimate. Default 0.1.
                Smaller values require more items.
            ability_range: (min, max) theta range to evaluate over.
            mean_discrimination: Expected average item discrimination.
            mean_difficulty: Expected average item difficulty.

        Returns:
            PowerResult with minimum number of items needed.
        """
        if target_se <= 0:
            raise ValueError(f"target_se must be positive, got {target_se}.")

        # Compute mean information per item across the ability range
        theta = np.linspace(ability_range[0], ability_range[1], 101)
        p = 1.0 / (1.0 + np.exp(-mean_discrimination * (theta - mean_difficulty)))
        mean_info_per_item = np.mean(mean_discrimination**2 * p * (1 - p))

        # SE(θ) = 1 / sqrt(n_items * I_per_item)
        # n_items = 1 / (SE² * I_per_item)
        if mean_info_per_item <= 0:
            n_items = float("inf")
        else:
            n_items = 1.0 / (target_se**2 * mean_info_per_item)

        n_items_ceil = int(np.ceil(n_items))

        return PowerResult(
            metric="min_items",
            value=n_items_ceil,
            details=(
                f"target SE={target_se}, ability_range={ability_range}, "
                f"mean_a={mean_discrimination}, mean_b={mean_difficulty}, "
                f"mean_info_per_item={mean_info_per_item:.4f}"
            ),
        )

    def min_samples_for_difference(
        self,
        effect_size: float = 0.5,
        alpha: float = 0.05,
        power: float = 0.80,
        n_groups: int = 2,
    ) -> PowerResult:
        """Estimate minimum samples to detect a score difference between models.

        Uses the normal approximation for a two-sample or k-sample test.
        For comparing two models' benchmark scores with desired statistical power.

        Args:
            effect_size: Expected Cohen's d (standardized mean difference).
                0.2=small, 0.5=medium, 0.8=large.
            alpha: Significance level. Default 0.05.
            power: Desired statistical power (1 - β). Default 0.80.
            n_groups: Number of groups to compare. Default 2.

        Returns:
            PowerResult with minimum samples per group.
        """
        if effect_size <= 0:
            raise ValueError(f"effect_size must be positive, got {effect_size}.")
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}.")
        if not 0 < power < 1:
            raise ValueError(f"power must be between 0 and 1, got {power}.")

        # For two-sample t-test: n = (z_α/2 + z_β)² * 2 / d²
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        if n_groups == 2:
            n_per_group = (z_alpha + z_beta) ** 2 * 2 / effect_size**2
        else:
            # For k groups (ANOVA-like), approximate using f = d/2
            f_effect = effect_size / 2
            n_per_group = (z_alpha + z_beta) ** 2 / f_effect**2

        n_ceil = int(np.ceil(n_per_group))

        return PowerResult(
            metric="min_samples_per_group",
            value=n_ceil,
            details=(
                f"effect_size={effect_size}, alpha={alpha}, power={power}, "
                f"n_groups={n_groups}"
            ),
        )

    def bootstrap_ci(
        self,
        scores: np.ndarray | pd.Series,
        statistic: str = "mean",
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
        seed: int | None = 42,
    ) -> BootstrapCIResult:
        """Compute bootstrap confidence intervals for evaluation scores.

        Uses the bias-corrected and accelerated (BCa) bootstrap method
        via percentile approximation for robust confidence intervals.

        Args:
            scores: Array of evaluation scores.
            statistic: Summary statistic to bootstrap.
                'mean', 'median', or 'std'. Default 'mean'.
            n_bootstrap: Number of bootstrap resamples. Default 1000.
            ci_level: Confidence level. Default 0.95.
            seed: Random seed for reproducibility.

        Returns:
            BootstrapCIResult with point estimate and confidence interval.
        """
        scores = np.asarray(scores, dtype=float)
        scores = scores[~np.isnan(scores)]

        if len(scores) < 2:
            raise ValueError(
                f"Need at least 2 non-NaN scores, got {len(scores)}."
            )
        if not 0 < ci_level < 1:
            raise ValueError(f"ci_level must be between 0 and 1, got {ci_level}.")

        stat_fn = {
            "mean": np.mean,
            "median": np.median,
            "std": np.std,
        }.get(statistic)
        if stat_fn is None:
            raise ValueError(
                f"Unknown statistic '{statistic}'. Choose from: 'mean', 'median', 'std'."
            )

        rng = np.random.default_rng(seed)
        point_estimate = float(stat_fn(scores))

        # Bootstrap resampling
        boot_stats = np.empty(n_bootstrap)
        n = len(scores)
        for i in range(n_bootstrap):
            sample = scores[rng.integers(0, n, size=n)]
            boot_stats[i] = stat_fn(sample)

        # Percentile method for CI
        alpha = 1 - ci_level
        ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
        se = float(np.std(boot_stats, ddof=1))

        return BootstrapCIResult(
            estimate=point_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=ci_level,
            se=se,
            n_bootstrap=n_bootstrap,
        )
