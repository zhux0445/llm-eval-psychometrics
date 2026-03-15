"""Tests for the reliability module: agreement and power analysis."""

import numpy as np
import pandas as pd
import pytest

from llm_eval_psychometrics.reliability import AgreementAnalyzer, PowerCalculator
from llm_eval_psychometrics.reliability.agreement import AgreementResult
from llm_eval_psychometrics.reliability.power import BootstrapCIResult, PowerResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_agreement_wide():
    """Two raters in perfect agreement."""
    return pd.DataFrame({
        "rater_A": [1, 2, 3, 4, 5],
        "rater_B": [1, 2, 3, 4, 5],
    })


@pytest.fixture
def moderate_agreement_wide():
    """Two raters with moderate agreement."""
    rng = np.random.default_rng(42)
    base = rng.integers(1, 6, size=50)
    noise = rng.choice([-1, 0, 0, 0, 1], size=50)
    return pd.DataFrame({
        "rater_A": base,
        "rater_B": np.clip(base + noise, 1, 5),
    })


@pytest.fixture
def three_raters_wide():
    """Three raters with varying agreement."""
    rng = np.random.default_rng(42)
    base = rng.integers(1, 6, size=30)
    return pd.DataFrame({
        "rater_A": base,
        "rater_B": np.clip(base + rng.choice([-1, 0, 0, 1], size=30), 1, 5),
        "rater_C": np.clip(base + rng.choice([-1, 0, 1], size=30), 1, 5),
    })


@pytest.fixture
def long_format_ratings():
    """Ratings in long format (item_id, rater_id, score)."""
    rows = []
    rng = np.random.default_rng(42)
    for item in range(20):
        base = rng.integers(1, 6)
        for rater in ["A", "B"]:
            score = int(np.clip(base + rng.choice([-1, 0, 0, 1]), 1, 5))
            rows.append({"item_id": item, "rater_id": rater, "score": score})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# AgreementAnalyzer init tests
# ---------------------------------------------------------------------------

class TestAgreementAnalyzerInit:
    def test_wide_format(self, moderate_agreement_wide):
        analyzer = AgreementAnalyzer(moderate_agreement_wide)
        assert analyzer._n_raters == 2
        assert analyzer._n_items == 50

    def test_long_format(self, long_format_ratings):
        analyzer = AgreementAnalyzer(long_format_ratings)
        assert analyzer._n_raters == 2
        assert analyzer._n_items == 20

    def test_not_dataframe_raises(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            AgreementAnalyzer([[1, 2], [3, 4]])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            AgreementAnalyzer(pd.DataFrame())

    def test_single_rater_raises(self):
        with pytest.raises(ValueError, match="at least 2 raters"):
            AgreementAnalyzer(pd.DataFrame({"rater_A": [1, 2, 3]}))


# ---------------------------------------------------------------------------
# Cohen's Kappa tests
# ---------------------------------------------------------------------------

class TestCohensKappa:
    def test_perfect_agreement(self, perfect_agreement_wide):
        result = AgreementAnalyzer(perfect_agreement_wide).cohens_kappa()
        assert isinstance(result, AgreementResult)
        assert result.value == pytest.approx(1.0)
        assert "perfect" in result.interpretation

    def test_moderate_agreement_range(self, moderate_agreement_wide):
        result = AgreementAnalyzer(moderate_agreement_wide).cohens_kappa()
        assert 0 < result.value < 1

    def test_weighted_linear(self, moderate_agreement_wide):
        result = AgreementAnalyzer(moderate_agreement_wide).cohens_kappa(weighted="linear")
        assert isinstance(result.value, float)

    def test_weighted_quadratic(self, moderate_agreement_wide):
        result = AgreementAnalyzer(moderate_agreement_wide).cohens_kappa(weighted="quadratic")
        assert isinstance(result.value, float)

    def test_three_raters_pairwise(self, three_raters_wide):
        result = AgreementAnalyzer(three_raters_wide).cohens_kappa()
        assert "pairwise" in result.metric
        assert 0 < result.value < 1


# ---------------------------------------------------------------------------
# ICC tests
# ---------------------------------------------------------------------------

class TestICC:
    def test_perfect_agreement(self, perfect_agreement_wide):
        result = AgreementAnalyzer(perfect_agreement_wide).icc()
        assert result.value == pytest.approx(1.0)
        assert result.interpretation == "excellent"

    def test_returns_ci(self, moderate_agreement_wide):
        result = AgreementAnalyzer(moderate_agreement_wide).icc()
        assert result.ci_lower is not None
        assert result.ci_upper is not None
        assert result.ci_lower <= result.value <= result.ci_upper

    def test_model_types(self, moderate_agreement_wide):
        analyzer = AgreementAnalyzer(moderate_agreement_wide)
        for model in ["one-way-random", "two-way-random", "two-way-mixed"]:
            result = analyzer.icc(model=model)
            assert isinstance(result.value, float)
            assert -1 <= result.value <= 1

    def test_average_measure(self, moderate_agreement_wide):
        analyzer = AgreementAnalyzer(moderate_agreement_wide)
        single = analyzer.icc(measure="single")
        average = analyzer.icc(measure="average")
        # Average measure ICC should be >= single measure
        assert average.value >= single.value

    def test_three_raters(self, three_raters_wide):
        result = AgreementAnalyzer(three_raters_wide).icc()
        assert isinstance(result.value, float)


# ---------------------------------------------------------------------------
# Krippendorff's alpha tests
# ---------------------------------------------------------------------------

class TestKrippendorffAlpha:
    def test_perfect_agreement(self, perfect_agreement_wide):
        result = AgreementAnalyzer(perfect_agreement_wide).krippendorff_alpha()
        assert isinstance(result, AgreementResult)
        assert result.value == pytest.approx(1.0, abs=0.01)

    def test_moderate_agreement(self, moderate_agreement_wide):
        result = AgreementAnalyzer(moderate_agreement_wide).krippendorff_alpha()
        assert 0 < result.value < 1

    def test_measurement_levels(self, moderate_agreement_wide):
        analyzer = AgreementAnalyzer(moderate_agreement_wide)
        for level in ["nominal", "ordinal", "interval", "ratio"]:
            result = analyzer.krippendorff_alpha(level_of_measurement=level)
            assert isinstance(result.value, float)

    def test_three_raters(self, three_raters_wide):
        result = AgreementAnalyzer(three_raters_wide).krippendorff_alpha()
        assert isinstance(result.value, float)


# ---------------------------------------------------------------------------
# PowerCalculator: min_items_for_precision tests
# ---------------------------------------------------------------------------

class TestMinItems:
    def test_returns_power_result(self):
        result = PowerCalculator().min_items_for_precision(target_se=0.1)
        assert isinstance(result, PowerResult)
        assert result.value > 0

    def test_tighter_se_needs_more_items(self):
        calc = PowerCalculator()
        loose = calc.min_items_for_precision(target_se=0.5)
        tight = calc.min_items_for_precision(target_se=0.1)
        assert tight.value > loose.value

    def test_higher_discrimination_needs_fewer_items(self):
        calc = PowerCalculator()
        low_a = calc.min_items_for_precision(target_se=0.1, mean_discrimination=0.5)
        high_a = calc.min_items_for_precision(target_se=0.1, mean_discrimination=2.0)
        assert high_a.value < low_a.value

    def test_zero_se_raises(self):
        with pytest.raises(ValueError, match="positive"):
            PowerCalculator().min_items_for_precision(target_se=0)


# ---------------------------------------------------------------------------
# PowerCalculator: min_samples_for_difference tests
# ---------------------------------------------------------------------------

class TestMinSamples:
    def test_returns_power_result(self):
        result = PowerCalculator().min_samples_for_difference(effect_size=0.5)
        assert isinstance(result, PowerResult)
        assert result.value > 0

    def test_smaller_effect_needs_more_samples(self):
        calc = PowerCalculator()
        large = calc.min_samples_for_difference(effect_size=0.8)
        small = calc.min_samples_for_difference(effect_size=0.2)
        assert small.value > large.value

    def test_higher_power_needs_more_samples(self):
        calc = PowerCalculator()
        low_power = calc.min_samples_for_difference(effect_size=0.5, power=0.70)
        high_power = calc.min_samples_for_difference(effect_size=0.5, power=0.95)
        assert high_power.value > low_power.value

    def test_invalid_effect_size(self):
        with pytest.raises(ValueError, match="positive"):
            PowerCalculator().min_samples_for_difference(effect_size=0)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            PowerCalculator().min_samples_for_difference(effect_size=0.5, alpha=1.5)


# ---------------------------------------------------------------------------
# PowerCalculator: bootstrap_ci tests
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_returns_result(self):
        scores = np.array([3.0, 4.0, 5.0, 3.5, 4.5])
        result = PowerCalculator().bootstrap_ci(scores)
        assert isinstance(result, BootstrapCIResult)

    def test_ci_contains_estimate(self):
        rng = np.random.default_rng(42)
        scores = rng.normal(3.0, 1.0, size=100)
        result = PowerCalculator().bootstrap_ci(scores, n_bootstrap=2000)
        assert result.ci_lower <= result.estimate <= result.ci_upper

    def test_mean_statistic(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = PowerCalculator().bootstrap_ci(scores, statistic="mean")
        assert result.estimate == pytest.approx(3.0)

    def test_median_statistic(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        result = PowerCalculator().bootstrap_ci(scores, statistic="median")
        assert result.estimate == pytest.approx(3.0)

    def test_std_statistic(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = PowerCalculator().bootstrap_ci(scores, statistic="std")
        assert result.estimate > 0

    def test_invalid_statistic(self):
        with pytest.raises(ValueError, match="Unknown statistic"):
            PowerCalculator().bootstrap_ci(np.array([1, 2, 3]), statistic="mode")

    def test_too_few_scores(self):
        with pytest.raises(ValueError, match="at least 2"):
            PowerCalculator().bootstrap_ci(np.array([1.0]))

    def test_handles_nan(self):
        scores = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        result = PowerCalculator().bootstrap_ci(scores)
        assert result.estimate == pytest.approx(3.0)

    def test_pandas_series(self):
        scores = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = PowerCalculator().bootstrap_ci(scores)
        assert isinstance(result, BootstrapCIResult)

    def test_wider_ci_at_lower_level(self):
        rng = np.random.default_rng(42)
        scores = rng.normal(0, 1, size=50)
        calc = PowerCalculator()
        ci_95 = calc.bootstrap_ci(scores, ci_level=0.95)
        ci_80 = calc.bootstrap_ci(scores, ci_level=0.80)
        width_95 = ci_95.ci_upper - ci_95.ci_lower
        width_80 = ci_80.ci_upper - ci_80.ci_lower
        assert width_95 > width_80
