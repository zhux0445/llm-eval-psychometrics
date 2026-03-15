"""Tests for the IRT module."""

import numpy as np
import pandas as pd
import pytest

from llm_eval_psychometrics.irt import (
    IRTAnalyzer,
    IRTResults,
    classical_item_stats,
    diagnostic_report,
    item_fit_residuals,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_response_matrix(n_models: int = 30, n_items: int = 20, seed: int = 42) -> np.ndarray:
    """Generate a synthetic response matrix with realistic IRT structure.

    Creates data where higher-ability models have higher probability of
    answering correctly, and items vary in difficulty.
    """
    rng = np.random.default_rng(seed)
    # True abilities uniformly spread
    theta = np.linspace(-2, 2, n_models)
    # True difficulties uniformly spread
    b = np.linspace(-1.5, 1.5, n_items)
    # Discrimination all ~1
    a = rng.uniform(0.5, 2.0, size=n_items)

    # ICC: P = 1 / (1 + exp(-a*(theta - b)))
    prob = 1 / (1 + np.exp(-a[np.newaxis, :] * (theta[:, np.newaxis] - b[np.newaxis, :])))
    responses = (rng.random((n_models, n_items)) < prob).astype(int)
    return responses


@pytest.fixture
def response_matrix():
    return _make_response_matrix()


@pytest.fixture
def fitted_2pl(response_matrix):
    analyzer = IRTAnalyzer(model="2PL")
    return analyzer.fit(response_matrix)


# ---------------------------------------------------------------------------
# IRTAnalyzer tests
# ---------------------------------------------------------------------------

class TestIRTAnalyzer:
    def test_invalid_model(self):
        with pytest.raises(ValueError, match="Unsupported model"):
            IRTAnalyzer(model="4PL")

    def test_model_case_insensitive(self):
        analyzer = IRTAnalyzer(model="2pl")
        assert analyzer.model == "2PL"

    def test_fit_2pl_returns_irt_results(self, response_matrix):
        analyzer = IRTAnalyzer(model="2PL")
        results = analyzer.fit(response_matrix)
        assert isinstance(results, IRTResults)
        assert results.model_type == "2PL"

    def test_fit_2pl_shapes(self, response_matrix):
        n_models, n_items = response_matrix.shape
        analyzer = IRTAnalyzer(model="2PL")
        results = analyzer.fit(response_matrix)

        assert results.item_params.shape == (n_items, 2)  # discrimination, difficulty
        assert len(results.model_ability) == n_models
        assert "discrimination" in results.item_params.columns
        assert "difficulty" in results.item_params.columns

    def test_fit_3pl_has_guessing(self, response_matrix):
        analyzer = IRTAnalyzer(model="3PL")
        results = analyzer.fit(response_matrix)
        assert "guessing" in results.item_params.columns
        assert results.model_type == "3PL"
        # Guessing should be between 0 and 1
        assert (results.item_params["guessing"] >= 0).all()
        assert (results.item_params["guessing"] <= 1).all()

    def test_custom_ids(self, response_matrix):
        n_models, n_items = response_matrix.shape
        item_ids = [f"q{i}" for i in range(n_items)]
        model_ids = [f"llm_{i}" for i in range(n_models)]

        analyzer = IRTAnalyzer(model="2PL")
        results = analyzer.fit(response_matrix, item_ids=item_ids, model_ids=model_ids)

        assert list(results.item_params.index) == item_ids
        assert list(results.model_ability.index) == model_ids

    def test_wrong_item_ids_length(self, response_matrix):
        analyzer = IRTAnalyzer(model="2PL")
        with pytest.raises(ValueError, match="item_ids length"):
            analyzer.fit(response_matrix, item_ids=["a", "b"])

    def test_wrong_model_ids_length(self, response_matrix):
        analyzer = IRTAnalyzer(model="2PL")
        with pytest.raises(ValueError, match="model_ids length"):
            analyzer.fit(response_matrix, model_ids=["a"])

    def test_aic_bic_present_2pl(self, fitted_2pl):
        assert fitted_2pl.aic is not None
        assert fitted_2pl.bic is not None

    def test_aic_bic_none_3pl(self, response_matrix):
        analyzer = IRTAnalyzer(model="3PL")
        results = analyzer.fit(response_matrix)
        assert results.aic is None


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_non_binary_raises(self):
        bad = np.array([[0, 1, 2], [1, 0, 1]])
        with pytest.raises(ValueError, match="only 0, 1, or NaN"):
            IRTAnalyzer().fit(bad)

    def test_1d_raises(self):
        with pytest.raises(ValueError, match="must be 2D"):
            IRTAnalyzer().fit(np.array([1, 0, 1]))

    def test_single_model_raises(self):
        with pytest.raises(ValueError, match="at least 2 models"):
            IRTAnalyzer().fit(np.array([[1, 0, 1]]))

    def test_single_item_raises(self):
        with pytest.raises(ValueError, match="at least 2 items"):
            IRTAnalyzer().fit(np.array([[1], [0]]))


# ---------------------------------------------------------------------------
# IRTResults methods tests
# ---------------------------------------------------------------------------

class TestIRTResults:
    def test_flag_poor_items_returns_dataframe(self, fitted_2pl):
        flagged = fitted_2pl.flag_poor_items()
        assert isinstance(flagged, pd.DataFrame)
        assert "reasons" in flagged.columns or flagged.empty

    def test_flag_poor_items_strict_threshold(self, fitted_2pl):
        # With very strict thresholds, more items should be flagged
        strict = fitted_2pl.flag_poor_items(min_discrimination=5.0)
        lenient = fitted_2pl.flag_poor_items(min_discrimination=0.01)
        assert len(strict) >= len(lenient)

    def test_item_information_shape(self, fitted_2pl):
        theta = np.linspace(-3, 3, 61)
        info = fitted_2pl.item_information(theta)
        assert info.shape == (61, fitted_2pl.item_params.shape[0])
        # Information should be non-negative
        assert (info.values >= 0).all()

    def test_item_information_default_theta(self, fitted_2pl):
        info = fitted_2pl.item_information()
        assert info.shape[0] == 81  # default linspace(-4, 4, 81)

    def test_test_information_shape(self, fitted_2pl):
        tic = fitted_2pl.test_information()
        assert isinstance(tic, pd.Series)
        assert len(tic) == 81
        # Test information = sum of item info, should be > 0
        assert (tic.values >= 0).all()

    def test_test_information_is_sum_of_item_info(self, fitted_2pl):
        theta = np.linspace(-3, 3, 21)
        item_info = fitted_2pl.item_information(theta)
        tic = fitted_2pl.test_information(theta)
        np.testing.assert_allclose(tic.values, item_info.sum(axis=1).values, atol=1e-10)


# ---------------------------------------------------------------------------
# Diagnostics tests
# ---------------------------------------------------------------------------

class TestClassicalItemStats:
    def test_output_shape(self, response_matrix):
        stats = classical_item_stats(response_matrix)
        assert stats.shape == (response_matrix.shape[1], 3)
        assert set(stats.columns) == {"p_value", "point_biserial", "variance"}

    def test_p_value_range(self, response_matrix):
        stats = classical_item_stats(response_matrix)
        assert (stats["p_value"] >= 0).all()
        assert (stats["p_value"] <= 1).all()

    def test_all_correct_item(self):
        """An item answered correctly by all models should have p=1, zero variance."""
        mat = np.ones((10, 3))
        mat[:, 1] = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        stats = classical_item_stats(mat)
        assert stats.loc["item_0", "p_value"] == 1.0
        assert stats.loc["item_0", "variance"] == 0.0


class TestItemFitResiduals:
    def test_output_shape(self, fitted_2pl):
        fit = item_fit_residuals(fitted_2pl)
        assert fit.shape == (fitted_2pl.item_params.shape[0], 1)
        assert "rmsr" in fit.columns

    def test_rmsr_non_negative(self, fitted_2pl):
        fit = item_fit_residuals(fitted_2pl)
        assert (fit["rmsr"] >= 0).all()


class TestDiagnosticReport:
    def test_report_columns(self, fitted_2pl):
        report = diagnostic_report(fitted_2pl)
        expected_cols = {"discrimination", "difficulty", "p_value",
                         "point_biserial", "variance", "rmsr", "quality"}
        assert expected_cols.issubset(set(report.columns))

    def test_quality_values(self, fitted_2pl):
        report = diagnostic_report(fitted_2pl)
        assert set(report["quality"].unique()).issubset({"good", "acceptable", "poor"})

    def test_report_length(self, fitted_2pl):
        report = diagnostic_report(fitted_2pl)
        assert len(report) == fitted_2pl.item_params.shape[0]
