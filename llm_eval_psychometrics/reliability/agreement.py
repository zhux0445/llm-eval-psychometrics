"""Inter-rater agreement metrics for LLM evaluation reliability.

Provides Cohen's Kappa, Intraclass Correlation Coefficient (ICC),
and Krippendorff's alpha for assessing agreement among LLM judges
or human annotators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import krippendorff as kripp
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class AgreementResult:
    """Result of an agreement analysis.

    Attributes:
        metric: Name of the agreement metric.
        value: The computed agreement coefficient.
        ci_lower: Lower bound of 95% confidence interval (if available).
        ci_upper: Upper bound of 95% confidence interval (if available).
        interpretation: Qualitative interpretation of the value.
        details: Additional information.
    """

    metric: str
    value: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    interpretation: str = ""
    details: str = ""

    def __repr__(self) -> str:
        ci = ""
        if self.ci_lower is not None:
            ci = f", 95% CI [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
        return (
            f"AgreementResult({self.metric} = {self.value:.4f}{ci}, "
            f"{self.interpretation})"
        )


def _interpret_kappa(value: float) -> str:
    """Interpret Cohen's Kappa / Krippendorff's alpha using Landis & Koch scale."""
    if value < 0:
        return "poor (less than chance)"
    if value < 0.20:
        return "slight"
    if value < 0.40:
        return "fair"
    if value < 0.60:
        return "moderate"
    if value < 0.80:
        return "substantial"
    return "almost perfect"


def _interpret_icc(value: float) -> str:
    """Interpret ICC using Cicchetti (1994) guidelines."""
    if value < 0.40:
        return "poor"
    if value < 0.60:
        return "fair"
    if value < 0.75:
        return "good"
    return "excellent"


class AgreementAnalyzer:
    """Compute inter-rater agreement metrics for evaluation reliability.

    Args:
        ratings_df: DataFrame in one of two formats:

            **Wide format** (raters as columns):
                Each row is an item, each column is a rater's score.

            **Long format** (one row per rating):
                Must contain columns 'item_id', 'rater_id', and 'score'.

    Example:
        >>> import pandas as pd
        >>> # Wide format: rows=items, columns=raters
        >>> ratings = pd.DataFrame({
        ...     'rater_A': [4, 3, 5, 2, 4],
        ...     'rater_B': [3, 3, 4, 2, 5],
        ... })
        >>> agree = AgreementAnalyzer(ratings)
        >>> agree.cohens_kappa()
        >>> agree.icc()
        >>> agree.krippendorff_alpha()
    """

    def __init__(self, ratings_df: pd.DataFrame) -> None:
        if not isinstance(ratings_df, pd.DataFrame):
            raise TypeError(
                f"ratings_df must be a pandas DataFrame, got {type(ratings_df).__name__}."
            )
        if ratings_df.empty:
            raise ValueError("ratings_df is empty.")

        # Detect format and convert to wide matrix
        if {"item_id", "rater_id", "score"}.issubset(ratings_df.columns):
            # Long format -> pivot to wide
            self._wide = ratings_df.pivot(
                index="item_id", columns="rater_id", values="score"
            )
        else:
            # Assume wide format
            self._wide = ratings_df.copy()

        self._matrix = self._wide.values.astype(float)
        self._n_items, self._n_raters = self._matrix.shape

        if self._n_raters < 2:
            raise ValueError(
                f"Need at least 2 raters, got {self._n_raters}. "
                "If using long format, ensure 'rater_id' has >= 2 distinct values."
            )

    def cohens_kappa(self, weighted: Literal["linear", "quadratic"] | None = None) -> AgreementResult:
        """Compute Cohen's Kappa for two raters.

        Measures agreement beyond what would be expected by chance.
        For more than 2 raters, computes the average pairwise Kappa.

        Args:
            weighted: Weighting scheme for ordinal data.
                None = unweighted (nominal data).
                'linear' = linearly weighted (ordinal, penalizes proportionally).
                'quadratic' = quadratically weighted (ordinal, penalizes by squared distance).

        Returns:
            AgreementResult with Kappa value and interpretation.
        """
        if self._n_raters == 2:
            kappa = self._compute_kappa(
                self._matrix[:, 0], self._matrix[:, 1], weighted
            )
            return AgreementResult(
                metric="cohens_kappa",
                value=kappa,
                interpretation=_interpret_kappa(kappa),
                details=f"2 raters, {self._n_items} items, weighted={weighted}",
            )

        # Average pairwise kappa for >2 raters
        kappas = []
        for i in range(self._n_raters):
            for j in range(i + 1, self._n_raters):
                k = self._compute_kappa(
                    self._matrix[:, i], self._matrix[:, j], weighted
                )
                kappas.append(k)

        mean_kappa = float(np.mean(kappas))
        return AgreementResult(
            metric="cohens_kappa (pairwise mean)",
            value=mean_kappa,
            interpretation=_interpret_kappa(mean_kappa),
            details=(
                f"{self._n_raters} raters, {self._n_items} items, "
                f"{len(kappas)} pairs, weighted={weighted}"
            ),
        )

    @staticmethod
    def _compute_kappa(
        r1: np.ndarray,
        r2: np.ndarray,
        weighted: str | None,
    ) -> float:
        """Compute Cohen's Kappa between two rater arrays.

        Args:
            r1: Ratings from rater 1.
            r2: Ratings from rater 2.
            weighted: Weighting scheme.

        Returns:
            Kappa coefficient.
        """
        # Drop pairs where either rater has NaN
        valid = ~(np.isnan(r1) | np.isnan(r2))
        r1, r2 = r1[valid], r2[valid]
        if len(r1) == 0:
            return np.nan

        categories = np.unique(np.concatenate([r1, r2]))
        n = len(r1)
        k = len(categories)
        cat_idx = {c: i for i, c in enumerate(categories)}

        # Confusion matrix
        cm = np.zeros((k, k))
        for a, b in zip(r1, r2):
            cm[cat_idx[a], cat_idx[b]] += 1

        # Observed agreement
        if weighted is None:
            # Unweighted: proportion of exact agreements
            p_o = np.trace(cm) / n
            p_e = sum(
                (cm[i, :].sum() / n) * (cm[:, i].sum() / n) for i in range(k)
            )
        else:
            # Weighted kappa
            weights = np.zeros((k, k))
            for i in range(k):
                for j in range(k):
                    if weighted == "linear":
                        weights[i, j] = 1 - abs(i - j) / (k - 1) if k > 1 else 1
                    elif weighted == "quadratic":
                        weights[i, j] = 1 - (i - j) ** 2 / (k - 1) ** 2 if k > 1 else 1

            p_o = np.sum(weights * cm) / n
            outer = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / n**2
            p_e = np.sum(weights * outer)

        if p_e == 1.0:
            return 1.0 if p_o == 1.0 else 0.0
        return float((p_o - p_e) / (1 - p_e))

    def icc(
        self,
        model: Literal[
            "one-way-random",
            "two-way-random",
            "two-way-mixed",
        ] = "two-way-mixed",
        measure: Literal["single", "average"] = "single",
    ) -> AgreementResult:
        """Compute the Intraclass Correlation Coefficient (ICC).

        ICC measures the proportion of total variance attributable to
        true differences between items (vs. rater disagreement).

        Models (Shrout & Fleiss, 1979):
        - 'one-way-random': ICC(1,1) — each item rated by different random raters
        - 'two-way-random': ICC(2,1) — raters are a random sample, generalize to population
        - 'two-way-mixed': ICC(3,1) — the specific raters are of interest (most common for LLM judges)

        Args:
            model: ICC model type. Default 'two-way-mixed'.
            measure: 'single' for single-rater reliability, 'average' for
                mean of k raters. Default 'single'.

        Returns:
            AgreementResult with ICC value, 95% CI, and interpretation.
        """
        # Drop rows with any NaN
        data = self._matrix[~np.any(np.isnan(self._matrix), axis=1)]
        n = data.shape[0]
        k = data.shape[1]

        if n < 2:
            raise ValueError(
                f"Need at least 2 complete items (rows without NaN), got {n}."
            )

        # ANOVA decomposition
        grand_mean = data.mean()
        row_means = data.mean(axis=1)
        col_means = data.mean(axis=0)

        # Sum of squares
        ss_total = np.sum((data - grand_mean) ** 2)
        ss_rows = k * np.sum((row_means - grand_mean) ** 2)  # Between subjects
        ss_cols = n * np.sum((col_means - grand_mean) ** 2)  # Between raters
        ss_error = ss_total - ss_rows - ss_cols  # Residual

        # Mean squares
        ms_rows = ss_rows / (n - 1)
        ms_cols = ss_cols / (k - 1) if k > 1 else 0
        ms_error = ss_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 0
        ms_within = (ss_total - ss_rows) / (n * (k - 1)) if n * (k - 1) > 0 else 0

        # ICC computation
        if model == "one-way-random":
            # ICC(1,1)
            icc_val = (ms_rows - ms_within) / (ms_rows + (k - 1) * ms_within)
            icc_label = "ICC(1,1)"
        elif model == "two-way-random":
            # ICC(2,1)
            icc_val = (ms_rows - ms_error) / (
                ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
            )
            icc_label = "ICC(2,1)"
        else:  # two-way-mixed
            # ICC(3,1)
            icc_val = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error)
            icc_label = "ICC(3,1)"

        if measure == "average":
            # Spearman-Brown formula for k raters
            icc_val = k * icc_val / (1 + (k - 1) * icc_val)
            icc_label = icc_label.replace(",1)", f",k={k})")

        # F-test based confidence interval (McGraw & Wong, 1996)
        ci_lower, ci_upper = self._icc_ci(
            icc_val, n, k, ms_rows, ms_error, ms_cols, model, measure
        )

        return AgreementResult(
            metric=icc_label,
            value=float(icc_val),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            interpretation=_interpret_icc(icc_val),
            details=f"{n} items, {k} raters, model={model}, measure={measure}",
        )

    @staticmethod
    def _icc_ci(
        icc: float,
        n: int,
        k: int,
        ms_rows: float,
        ms_error: float,
        ms_cols: float,
        model: str,
        measure: str,
        alpha: float = 0.05,
    ) -> tuple[float, float]:
        """Compute F-test based CI for ICC.

        Uses the approach from McGraw & Wong (1996) for two-way models.
        """
        df1 = n - 1
        df2 = (n - 1) * (k - 1)
        f_lower = stats.f.ppf(1 - alpha / 2, df1, df2)
        f_upper = stats.f.ppf(alpha / 2, df1, df2)

        if ms_error == 0:
            return (icc, icc)

        f_obs = ms_rows / ms_error

        if model in ("two-way-mixed", "two-way-random"):
            # CI for ICC(3,1)
            lower = (f_obs / f_lower - 1) / (f_obs / f_lower + k - 1)
            upper = (f_obs / f_upper - 1) / (f_obs / f_upper + k - 1)
        else:
            # CI for ICC(1,1)
            lower = (f_obs / f_lower - 1) / (f_obs / f_lower + k - 1)
            upper = (f_obs / f_upper - 1) / (f_obs / f_upper + k - 1)

        if measure == "average":
            lower = k * lower / (1 + (k - 1) * lower)
            upper = k * upper / (1 + (k - 1) * upper)

        return (lower, upper)

    def krippendorff_alpha(
        self,
        level_of_measurement: Literal["nominal", "ordinal", "interval", "ratio"] = "interval",
    ) -> AgreementResult:
        """Compute Krippendorff's alpha reliability coefficient.

        A versatile agreement measure that handles any number of raters,
        missing data, and different measurement levels. Well-suited for
        LLM evaluation where raters and scales vary.

        Guidelines (Krippendorff, 2004):
        - α >= 0.80: reliable
        - 0.67 <= α < 0.80: tentatively acceptable
        - α < 0.67: unreliable

        Args:
            level_of_measurement: Scale type of the ratings.
                'nominal': unordered categories
                'ordinal': ordered categories
                'interval': equal intervals (default, most common for scores)
                'ratio': has a true zero point

        Returns:
            AgreementResult with alpha value and interpretation.
        """
        # krippendorff library expects [raters x items] with np.nan for missing
        reliability_data = self._matrix.T  # transpose to [raters x items]

        alpha_val = kripp.alpha(
            reliability_data=reliability_data,
            level_of_measurement=level_of_measurement,
        )

        return AgreementResult(
            metric="krippendorff_alpha",
            value=float(alpha_val),
            interpretation=_interpret_kappa(alpha_val),
            details=(
                f"{self._n_items} items, {self._n_raters} raters, "
                f"level={level_of_measurement}"
            ),
        )
