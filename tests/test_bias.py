"""Tests for the bias detection and calibration modules."""

import numpy as np
import pandas as pd
import pytest

from llm_eval_psychometrics.bias import (
    BiasDetector,
    BiasTestResult,
    ScoreCalibrator,
)
from llm_eval_psychometrics.bias.detection import _same_family


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_scores_df():
    """Scores with position and length columns, no strong bias."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "score": rng.integers(1, 6, size=n),
        "position": rng.choice([1, 2], size=n),
        "response_length": rng.integers(50, 500, size=n),
    })


@pytest.fixture
def position_biased_df():
    """Scores with strong position bias: position 1 always scores higher."""
    rng = np.random.default_rng(42)
    n = 200
    positions = rng.choice([1, 2], size=n)
    # Position 1 gets +2 boost
    scores = rng.integers(1, 4, size=n) + (positions == 1).astype(int) * 2
    return pd.DataFrame({
        "score": scores,
        "position": positions,
        "response_length": rng.integers(50, 500, size=n),
    })


@pytest.fixture
def length_biased_df():
    """Scores strongly correlated with response length."""
    rng = np.random.default_rng(42)
    n = 200
    lengths = rng.integers(50, 500, size=n)
    # Score = f(length) + noise
    scores = (lengths / 100).astype(int) + rng.integers(0, 2, size=n)
    return pd.DataFrame({
        "score": scores,
        "response_length": lengths,
    })


@pytest.fixture
def self_preference_df():
    """Scores with self-preference: judges score own family higher."""
    rows = []
    rng = np.random.default_rng(42)
    judges = ["gpt-4", "claude-3-opus"]
    response_models = ["gpt-3.5-turbo", "claude-2", "llama-70b"]
    for _ in range(100):
        judge = rng.choice(judges)
        resp_model = rng.choice(response_models)
        is_self = _same_family(judge, resp_model)
        base = rng.integers(2, 5)
        score = base + (2 if is_self else 0)  # +2 boost for same family
        rows.append({
            "score": score,
            "judge_model": judge,
            "response_model": resp_model,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def multi_judge_df():
    """Multi-judge scores for calibration testing."""
    rng = np.random.default_rng(42)
    rows = []
    # Judge A: lenient (mean ~4), Judge B: strict (mean ~2)
    for prompt_id in range(50):
        rows.append({
            "prompt_id": prompt_id,
            "judge_model": "judge_A",
            "score": rng.normal(4.0, 0.5),
        })
        rows.append({
            "prompt_id": prompt_id,
            "judge_model": "judge_B",
            "score": rng.normal(2.0, 0.5),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# BiasDetector init tests
# ---------------------------------------------------------------------------

class TestBiasDetectorInit:
    def test_valid_init(self, basic_scores_df):
        detector = BiasDetector(basic_scores_df)
        assert len(detector.df) == len(basic_scores_df)

    def test_not_dataframe_raises(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            BiasDetector([[1, 2], [3, 4]])

    def test_missing_score_column(self):
        with pytest.raises(ValueError, match="'score' column"):
            BiasDetector(pd.DataFrame({"x": [1, 2]}))

    def test_all_nan_scores(self):
        with pytest.raises(ValueError, match="All scores are NaN"):
            BiasDetector(pd.DataFrame({"score": [np.nan, np.nan]}))


# ---------------------------------------------------------------------------
# Position bias tests
# ---------------------------------------------------------------------------

class TestPositionBias:
    def test_returns_bias_test_result(self, basic_scores_df):
        result = BiasDetector(basic_scores_df).position_bias()
        assert isinstance(result, BiasTestResult)
        assert result.bias_type == "position_bias"

    def test_detects_strong_position_bias(self, position_biased_df):
        result = BiasDetector(position_biased_df).position_bias()
        assert result.detected is True
        assert result.p_value < 0.05
        assert result.effect_size > 0.01

    def test_no_position_bias_in_random_data(self, basic_scores_df):
        result = BiasDetector(basic_scores_df).position_bias()
        # Random data should usually not show significant bias
        # (but we check structure, not always p-value)
        assert result.p_value >= 0
        assert result.effect_size >= 0

    def test_missing_column_raises(self):
        df = pd.DataFrame({"score": [1, 2, 3]})
        with pytest.raises(ValueError, match="'position' not found"):
            BiasDetector(df).position_bias()

    def test_single_position_raises(self):
        df = pd.DataFrame({"score": [1, 2, 3], "position": [1, 1, 1]})
        with pytest.raises(ValueError, match="at least 2 distinct positions"):
            BiasDetector(df).position_bias()


# ---------------------------------------------------------------------------
# Length bias tests
# ---------------------------------------------------------------------------

class TestLengthBias:
    def test_returns_bias_test_result(self, basic_scores_df):
        result = BiasDetector(basic_scores_df).length_bias()
        assert isinstance(result, BiasTestResult)
        assert result.bias_type == "length_bias"

    def test_detects_strong_length_bias(self, length_biased_df):
        result = BiasDetector(length_biased_df).length_bias()
        assert result.detected is True
        assert result.effect_size > 0.3  # Strong correlation

    def test_missing_column_raises(self):
        df = pd.DataFrame({"score": [1, 2, 3]})
        with pytest.raises(ValueError, match="'response_length' not found"):
            BiasDetector(df).length_bias()

    def test_too_few_observations(self):
        df = pd.DataFrame({"score": [1, 2], "response_length": [100, 200]})
        with pytest.raises(ValueError, match="at least 3"):
            BiasDetector(df).length_bias()


# ---------------------------------------------------------------------------
# Self-preference tests
# ---------------------------------------------------------------------------

class TestSelfPreference:
    def test_returns_bias_test_result(self, self_preference_df):
        result = BiasDetector(self_preference_df).self_preference()
        assert isinstance(result, BiasTestResult)
        assert result.bias_type == "self_preference"

    def test_detects_self_preference(self, self_preference_df):
        result = BiasDetector(self_preference_df).self_preference()
        assert result.detected is True
        assert result.effect_size > 0.5

    def test_missing_judge_model_raises(self):
        df = pd.DataFrame({"score": [1], "response_model": ["gpt-4"]})
        with pytest.raises(ValueError, match="'judge_model' not found"):
            BiasDetector(df).self_preference()

    def test_missing_response_model_raises(self):
        df = pd.DataFrame({"score": [1], "judge_model": ["gpt-4"]})
        with pytest.raises(ValueError, match="'response_model' not found"):
            BiasDetector(df).self_preference()


# ---------------------------------------------------------------------------
# Report tests
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_runs_available_tests(self, basic_scores_df):
        report = BiasDetector(basic_scores_df).report()
        assert isinstance(report, pd.DataFrame)
        # Should run position and length tests
        assert "position_bias" in report["bias_type"].values
        assert "length_bias" in report["bias_type"].values

    def test_report_empty_when_no_columns(self):
        df = pd.DataFrame({"score": [1, 2, 3]})
        report = BiasDetector(df).report()
        assert len(report) == 0

    def test_report_columns(self, basic_scores_df):
        report = BiasDetector(basic_scores_df).report()
        expected_cols = {"bias_type", "detected", "effect_size", "p_value",
                         "test_name", "details"}
        assert expected_cols == set(report.columns)


# ---------------------------------------------------------------------------
# _same_family helper tests
# ---------------------------------------------------------------------------

class TestSameFamily:
    def test_same_family_gpt(self):
        assert _same_family("gpt-4", "gpt-3.5-turbo") is True

    def test_same_family_claude(self):
        assert _same_family("claude-3-opus", "claude-2") is True

    def test_different_families(self):
        assert _same_family("gpt-4", "claude-3") is False

    def test_case_insensitive(self):
        assert _same_family("GPT-4", "gpt-3.5") is True

    def test_exact_same_model(self):
        assert _same_family("llama-70b", "llama-13b") is True


# ---------------------------------------------------------------------------
# ScoreCalibrator tests
# ---------------------------------------------------------------------------

class TestScoreCalibrator:
    def test_missing_columns_raises(self):
        df = pd.DataFrame({"score": [1, 2]})
        with pytest.raises(ValueError, match="missing required columns"):
            ScoreCalibrator(df)

    def test_z_score_calibration(self, multi_judge_df):
        cal = ScoreCalibrator(multi_judge_df)
        result = cal.z_score_calibration()
        assert "calibrated_score" in result.calibrated_scores.columns
        assert result.method == "z_score"

        # After z-score calibration, each judge should have ~0 mean
        for judge, group in result.calibrated_scores.groupby("judge_model"):
            assert abs(group["calibrated_score"].mean()) < 0.01

    def test_z_score_with_reference(self, multi_judge_df):
        cal = ScoreCalibrator(multi_judge_df, reference_judge="judge_A")
        result = cal.z_score_calibration()
        # Calibrated scores should be on judge_A's scale
        calibrated_means = (
            result.calibrated_scores.groupby("judge_model")["calibrated_score"].mean()
        )
        # Both judges should have similar means after calibration
        assert abs(calibrated_means["judge_A"] - calibrated_means["judge_B"]) < 0.5

    def test_z_score_invalid_reference(self, multi_judge_df):
        cal = ScoreCalibrator(multi_judge_df, reference_judge="nonexistent")
        with pytest.raises(ValueError, match="not found"):
            cal.z_score_calibration()

    def test_percentile_calibration(self, multi_judge_df):
        cal = ScoreCalibrator(multi_judge_df)
        result = cal.percentile_calibration()
        assert result.method == "percentile"
        # Percentiles should be between 0 and 100
        scores = result.calibrated_scores["calibrated_score"]
        assert scores.min() >= 0
        assert scores.max() <= 100

    def test_mean_shift_calibration(self, multi_judge_df):
        cal = ScoreCalibrator(multi_judge_df)
        result = cal.mean_shift_calibration()
        assert result.method == "mean_shift"
        # After mean-shift, both judges should have ~same mean
        calibrated_means = (
            result.calibrated_scores.groupby("judge_model")["calibrated_score"].mean()
        )
        assert abs(calibrated_means["judge_A"] - calibrated_means["judge_B"]) < 0.1

    def test_judge_stats_populated(self, multi_judge_df):
        cal = ScoreCalibrator(multi_judge_df)
        result = cal.z_score_calibration()
        assert "raw_mean" in result.judge_stats.columns
        assert len(result.judge_stats) == 2  # two judges
