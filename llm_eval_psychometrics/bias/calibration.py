"""Cross-model score calibration for LLM-as-judge systems.

When multiple LLM judges evaluate the same responses, their raw scores
may not be directly comparable due to different scoring tendencies
(leniency, severity, scale usage). This module provides methods to
calibrate scores onto a common scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


@dataclass
class CalibrationResult:
    """Result of cross-model score calibration.

    Attributes:
        calibrated_scores: DataFrame with original and calibrated score columns.
        judge_stats: DataFrame of per-judge statistics (mean, std, n).
        method: Calibration method used.
    """

    calibrated_scores: pd.DataFrame
    judge_stats: pd.DataFrame
    method: str


class ScoreCalibrator:
    """Calibrate scores from multiple LLM judges onto a common scale.

    Args:
        scores_df: DataFrame with columns:
            - 'judge_model': str identifying the judge
            - 'score': numeric score
            - 'prompt_id' or 'response_id': identifier linking scores
              to the same evaluation target (needed for equating methods)
        reference_judge: Optional judge model name to use as reference
            for calibration. If None, calibrates to the grand mean.

    Example:
        >>> calibrator = ScoreCalibrator(scores_df)
        >>> result = calibrator.z_score_calibration()
        >>> print(result.calibrated_scores.head())
    """

    def __init__(
        self,
        scores_df: pd.DataFrame,
        reference_judge: str | None = None,
    ) -> None:
        required = {"judge_model", "score"}
        missing = required - set(scores_df.columns)
        if missing:
            raise ValueError(
                f"scores_df missing required columns: {missing}. "
                f"Available: {list(scores_df.columns)}"
            )

        self.df = scores_df.copy()
        self.reference_judge = reference_judge

        # Compute per-judge statistics
        self._judge_stats = (
            self.df.groupby("judge_model")["score"]
            .agg(["mean", "std", "count"])
            .rename(columns={"mean": "raw_mean", "std": "raw_std", "count": "n"})
        )
        # Fill zero std with 1 to avoid division by zero
        self._judge_stats["raw_std"] = self._judge_stats["raw_std"].replace(0, 1.0)

    def z_score_calibration(self) -> CalibrationResult:
        """Calibrate scores using z-score standardization.

        Each judge's scores are standardized to zero mean and unit variance,
        then optionally rescaled to a reference judge's distribution.

        This removes differences in leniency (mean) and spread (std)
        across judges, making scores comparable.

        Returns:
            CalibrationResult with calibrated scores.
        """
        result_df = self.df.copy()
        judge_stats = self._judge_stats.copy()

        # z-standardize each judge's scores
        result_df["calibrated_score"] = result_df.apply(
            lambda row: (
                (row["score"] - judge_stats.loc[row["judge_model"], "raw_mean"])
                / judge_stats.loc[row["judge_model"], "raw_std"]
            ),
            axis=1,
        )

        if self.reference_judge is not None:
            if self.reference_judge not in judge_stats.index:
                raise ValueError(
                    f"Reference judge '{self.reference_judge}' not found. "
                    f"Available judges: {list(judge_stats.index)}"
                )
            # Rescale to reference judge's distribution
            ref_mean = judge_stats.loc[self.reference_judge, "raw_mean"]
            ref_std = judge_stats.loc[self.reference_judge, "raw_std"]
            result_df["calibrated_score"] = (
                result_df["calibrated_score"] * ref_std + ref_mean
            )

        return CalibrationResult(
            calibrated_scores=result_df,
            judge_stats=judge_stats,
            method="z_score",
        )

    def percentile_calibration(self) -> CalibrationResult:
        """Calibrate scores using percentile rank transformation.

        Each judge's scores are converted to percentile ranks (0-100),
        eliminating differences in scale usage and distribution shape.

        This is more robust than z-score calibration when judges use
        different parts of the scale (e.g., one uses 1-3, another 3-5).

        Returns:
            CalibrationResult with calibrated scores (as percentiles 0-100).
        """
        result_df = self.df.copy()
        calibrated = np.zeros(len(result_df))

        for judge, group in result_df.groupby("judge_model"):
            idx = group.index
            # Percentile rank within this judge's score distribution
            ranks = sp_stats.rankdata(group["score"], method="average")
            percentiles = (ranks - 0.5) / len(ranks) * 100
            calibrated[idx] = percentiles

        result_df["calibrated_score"] = calibrated

        return CalibrationResult(
            calibrated_scores=result_df,
            judge_stats=self._judge_stats.copy(),
            method="percentile",
        )

    def mean_shift_calibration(self) -> CalibrationResult:
        """Calibrate scores by shifting each judge's mean to the grand mean.

        A minimal calibration that only corrects for leniency/severity
        differences, preserving each judge's variance and distribution shape.

        Appropriate when judges use similar scales but differ in overall
        strictness.

        Returns:
            CalibrationResult with calibrated scores.
        """
        result_df = self.df.copy()
        judge_stats = self._judge_stats.copy()
        grand_mean = self.df["score"].mean()

        result_df["calibrated_score"] = result_df.apply(
            lambda row: (
                row["score"]
                - judge_stats.loc[row["judge_model"], "raw_mean"]
                + grand_mean
            ),
            axis=1,
        )

        judge_stats["shift"] = grand_mean - judge_stats["raw_mean"]

        return CalibrationResult(
            calibrated_scores=result_df,
            judge_stats=judge_stats,
            method="mean_shift",
        )
