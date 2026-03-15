"""Bias detection for LLM-as-judge evaluation systems.

Detects systematic biases in LLM judge scores:
- Position bias: scores influenced by option/response ordering
- Length bias: scores correlated with response length
- Self-preference: judges favoring responses from same model family
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class BiasTestResult:
    """Result of a single bias test.

    Attributes:
        bias_type: Name of the bias tested.
        detected: Whether statistically significant bias was found.
        effect_size: Magnitude of the bias effect.
        p_value: Statistical significance (two-sided).
        statistic: The test statistic value.
        test_name: Name of the statistical test used.
        details: Additional information about the test result.
    """

    bias_type: str
    detected: bool
    effect_size: float
    p_value: float
    statistic: float
    test_name: str
    details: str

    def __repr__(self) -> str:
        status = "DETECTED" if self.detected else "not detected"
        return (
            f"BiasTestResult({self.bias_type}: {status}, "
            f"effect={self.effect_size:.4f}, p={self.p_value:.4f})"
        )


class BiasDetector:
    """Detect systematic biases in LLM-as-judge evaluation scores.

    Analyzes a DataFrame of judge scores to identify position bias,
    length bias, and self-preference bias using statistical tests.

    Args:
        scores_df: DataFrame with judge scoring data. Required columns:
            - 'score': numeric judge score
            At least one of the following for specific tests:
            - 'position': int position/order of the response (for position bias)
            - 'response_length': int character/token count (for length bias)
            - 'judge_model': str judge model name (for self-preference)
            - 'response_model': str response author model (for self-preference)
        alpha: Significance level for hypothesis tests. Default 0.05.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'score': [4, 3, 5, 2, 4, 3],
        ...     'position': [1, 2, 1, 2, 1, 2],
        ...     'response_length': [100, 200, 150, 80, 300, 120],
        ... })
        >>> detector = BiasDetector(df)
        >>> detector.position_bias()
    """

    REQUIRED_COLUMN = "score"

    def __init__(self, scores_df: pd.DataFrame, alpha: float = 0.05) -> None:
        if not isinstance(scores_df, pd.DataFrame):
            raise TypeError(
                f"scores_df must be a pandas DataFrame, got {type(scores_df).__name__}."
            )
        if self.REQUIRED_COLUMN not in scores_df.columns:
            raise ValueError(
                f"scores_df must contain a '{self.REQUIRED_COLUMN}' column. "
                f"Found columns: {list(scores_df.columns)}"
            )
        if scores_df["score"].isna().all():
            raise ValueError("All scores are NaN. Cannot perform bias analysis.")

        self.df = scores_df.copy()
        self.alpha = alpha

    def position_bias(self) -> BiasTestResult:
        """Test whether judge scores are influenced by response position.

        Uses the Kruskal-Wallis H-test (non-parametric one-way ANOVA) to
        compare score distributions across positions. This is robust to
        non-normal score distributions common in LLM evaluations.

        Effect size is eta-squared (η²): proportion of variance explained
        by position. η² > 0.06 is considered a medium effect.

        Returns:
            BiasTestResult with position bias analysis.

        Raises:
            ValueError: If 'position' column is missing or has < 2 levels.
        """
        self._check_column("position")
        df = self.df.dropna(subset=["score", "position"])

        groups = [g["score"].values for _, g in df.groupby("position")]
        if len(groups) < 2:
            raise ValueError(
                "Need at least 2 distinct positions to test position bias. "
                f"Found {len(groups)} position(s)."
            )

        # Kruskal-Wallis: non-parametric test for differences across groups
        h_stat, p_value = stats.kruskal(*groups)

        # Effect size: eta-squared = (H - k + 1) / (n - k)
        n = len(df)
        k = len(groups)
        eta_sq = max(0.0, (h_stat - k + 1) / (n - k)) if n > k else 0.0

        # Per-position mean scores for details
        pos_means = df.groupby("position")["score"].mean()
        means_str = ", ".join(f"pos {p}: {m:.3f}" for p, m in pos_means.items())

        return BiasTestResult(
            bias_type="position_bias",
            detected=bool(p_value < self.alpha),
            effect_size=eta_sq,
            p_value=p_value,
            statistic=h_stat,
            test_name="Kruskal-Wallis H-test",
            details=f"Mean scores by position: {means_str}",
        )

    def length_bias(self) -> BiasTestResult:
        """Test whether judge scores correlate with response length.

        Uses Spearman rank correlation to detect monotonic relationships
        between response length and scores, without assuming linearity.

        Effect size is |ρ| (absolute Spearman correlation).
        |ρ| > 0.3 is considered a medium effect.

        Returns:
            BiasTestResult with length bias analysis.

        Raises:
            ValueError: If 'response_length' column is missing.
        """
        self._check_column("response_length")
        df = self.df.dropna(subset=["score", "response_length"])

        if len(df) < 3:
            raise ValueError(
                f"Need at least 3 observations for correlation, got {len(df)}."
            )

        rho, p_value = stats.spearmanr(df["response_length"], df["score"])

        direction = "positive" if rho > 0 else "negative"
        return BiasTestResult(
            bias_type="length_bias",
            detected=bool(p_value < self.alpha),
            effect_size=abs(rho),
            p_value=p_value,
            statistic=rho,
            test_name="Spearman rank correlation",
            details=(
                f"ρ = {rho:.4f} ({direction}): "
                f"{'longer responses score higher' if rho > 0 else 'shorter responses score higher'}"
            ),
        )

    def self_preference(self) -> BiasTestResult:
        """Test whether judges prefer responses from their own model family.

        Compares scores given to "self" responses (judge and response from
        same model family) vs "other" responses using Mann-Whitney U test.

        Model family is determined by matching the prefix before common
        separators (e.g., 'gpt-4' and 'gpt-3.5' share family 'gpt').

        Effect size is Cohen's d (standardized mean difference).
        |d| > 0.5 is considered a medium effect.

        Returns:
            BiasTestResult with self-preference analysis.

        Raises:
            ValueError: If 'judge_model' or 'response_model' columns are missing,
                or if no self/other pairs exist.
        """
        self._check_column("judge_model")
        self._check_column("response_model")
        df = self.df.dropna(subset=["score", "judge_model", "response_model"])

        # Determine if judge and response are from the same model family
        df = df.copy()
        df["is_self"] = df.apply(
            lambda row: _same_family(row["judge_model"], row["response_model"]),
            axis=1,
        )

        self_scores = df.loc[df["is_self"], "score"].values
        other_scores = df.loc[~df["is_self"], "score"].values

        if len(self_scores) == 0 or len(other_scores) == 0:
            raise ValueError(
                "Need both self-model and other-model score pairs to test "
                "self-preference. Check that judge_model and response_model "
                "columns have overlapping model families."
            )

        # Mann-Whitney U: non-parametric comparison of two groups
        u_stat, p_value = stats.mannwhitneyu(
            self_scores, other_scores, alternative="two-sided"
        )

        # Cohen's d effect size
        pooled_std = np.sqrt(
            (np.var(self_scores, ddof=1) * (len(self_scores) - 1)
             + np.var(other_scores, ddof=1) * (len(other_scores) - 1))
            / (len(self_scores) + len(other_scores) - 2)
        )
        cohens_d = (
            (np.mean(self_scores) - np.mean(other_scores)) / pooled_std
            if pooled_std > 0
            else 0.0
        )

        return BiasTestResult(
            bias_type="self_preference",
            detected=bool(p_value < self.alpha),
            effect_size=abs(cohens_d),
            p_value=p_value,
            statistic=u_stat,
            test_name="Mann-Whitney U test",
            details=(
                f"Self mean: {np.mean(self_scores):.3f} (n={len(self_scores)}), "
                f"Other mean: {np.mean(other_scores):.3f} (n={len(other_scores)}), "
                f"Cohen's d: {cohens_d:+.3f}"
            ),
        )

    def report(self) -> pd.DataFrame:
        """Run all applicable bias tests and return a summary report.

        Automatically detects which tests can be run based on available
        columns in the DataFrame.

        Returns:
            DataFrame with one row per test, columns: bias_type, detected,
            effect_size, p_value, test_name, details.
        """
        results = []

        # Run each test if required columns exist
        tests = [
            ("position", self.position_bias),
            ("response_length", self.length_bias),
        ]
        for col, test_fn in tests:
            if col in self.df.columns:
                try:
                    results.append(test_fn())
                except ValueError:
                    pass

        # Self-preference needs two columns
        if "judge_model" in self.df.columns and "response_model" in self.df.columns:
            try:
                results.append(self.self_preference())
            except ValueError:
                pass

        if not results:
            return pd.DataFrame(
                columns=["bias_type", "detected", "effect_size", "p_value",
                         "test_name", "details"]
            )

        return pd.DataFrame(
            [
                {
                    "bias_type": r.bias_type,
                    "detected": r.detected,
                    "effect_size": r.effect_size,
                    "p_value": r.p_value,
                    "test_name": r.test_name,
                    "details": r.details,
                }
                for r in results
            ]
        )

    def _check_column(self, col: str) -> None:
        """Verify a required column exists in the DataFrame.

        Args:
            col: Column name to check.

        Raises:
            ValueError: If column is not present.
        """
        if col not in self.df.columns:
            raise ValueError(
                f"Column '{col}' not found in scores_df. "
                f"Available columns: {list(self.df.columns)}. "
                f"This column is required for the requested bias test."
            )


def _same_family(model_a: str, model_b: str) -> bool:
    """Determine if two model names belong to the same family.

    Uses prefix matching: extracts the family name by splitting on
    common separators ('-', '_', ' ') and comparing the first token.

    Args:
        model_a: First model name.
        model_b: Second model name.

    Returns:
        True if models share the same family prefix.

    Examples:
        >>> _same_family("gpt-4", "gpt-3.5-turbo")
        True
        >>> _same_family("claude-3-opus", "claude-2")
        True
        >>> _same_family("gpt-4", "claude-3")
        False
    """
    import re

    def _extract_family(name: str) -> str:
        # Split on hyphens, underscores, spaces, or digits
        # Take the first alphabetic token as the family name
        tokens = re.split(r"[-_\s]", name.strip().lower())
        return tokens[0] if tokens else name.lower()

    return _extract_family(model_a) == _extract_family(model_b)
