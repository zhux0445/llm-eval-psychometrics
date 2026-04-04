"""Tests for the demand profiling module."""

import numpy as np
import pandas as pd
import pytest

from llm_eval_psychometrics.demand import (
    DemandProfiler,
    DemandProfile,
    BenchmarkValidity,
    plot_demand_profile,
    plot_ability_vs_demand,
    plot_sensitivity_specificity,
)


@pytest.fixture
def sample_demand_matrix():
    """Demand matrix for 20 items across 5 dimensions."""
    np.random.seed(42)
    n_items = 20
    dims = [
        "logical_reasoning",
        "quantitative_reasoning",
        "verbal_comprehension",
        "knowledge_formal",
        "atypicality",
    ]
    data = {}
    # logical_reasoning: target, high demand
    data["logical_reasoning"] = np.random.choice([0, 1, 2, 3, 4, 5], n_items, p=[0.05, 0.1, 0.2, 0.3, 0.2, 0.15])
    # quantitative_reasoning: target, moderate demand
    data["quantitative_reasoning"] = np.random.choice([0, 1, 2, 3, 4], n_items, p=[0.1, 0.2, 0.3, 0.25, 0.15])
    # verbal_comprehension: low demand (not a target for math benchmark)
    data["verbal_comprehension"] = np.random.choice([0, 1, 2], n_items, p=[0.5, 0.3, 0.2])
    # knowledge_formal: moderate
    data["knowledge_formal"] = np.random.choice([0, 1, 2, 3], n_items, p=[0.2, 0.3, 0.3, 0.2])
    # atypicality: extraneous
    data["atypicality"] = np.random.choice([0, 1], n_items, p=[0.7, 0.3])

    item_ids = [f"item_{i}" for i in range(n_items)]
    return pd.DataFrame(data, index=item_ids)


@pytest.fixture
def sample_response_matrix():
    """Binary response matrix: 5 models x 20 items."""
    np.random.seed(123)
    return (np.random.rand(5, 20) > 0.4).astype(float)


@pytest.fixture
def sample_model_ids():
    return ["GPT-4", "Claude-3", "LLaMA-70B", "Mistral-7B", "Gemini"]


class TestDemandProfilerInit:
    def test_basic_init(self, sample_demand_matrix):
        profiler = DemandProfiler(sample_demand_matrix)
        assert profiler.n_items == 20
        assert len(profiler.dimensions) == 5

    def test_with_response_matrix(self, sample_demand_matrix, sample_response_matrix, sample_model_ids):
        profiler = DemandProfiler(
            sample_demand_matrix,
            response_matrix=sample_response_matrix,
            model_ids=sample_model_ids,
        )
        assert profiler.n_models == 5
        assert profiler.model_ids == sample_model_ids

    def test_invalid_demand_matrix(self):
        with pytest.raises(TypeError):
            DemandProfiler(np.array([[1, 2], [3, 4]]))

    def test_empty_demand_matrix(self):
        with pytest.raises(ValueError):
            DemandProfiler(pd.DataFrame())

    def test_mismatched_items(self, sample_demand_matrix):
        bad_responses = np.random.rand(3, 10)  # 10 items, but demand has 20
        with pytest.raises(ValueError, match="items"):
            DemandProfiler(sample_demand_matrix, response_matrix=bad_responses)

    def test_mismatched_model_ids(self, sample_demand_matrix, sample_response_matrix):
        with pytest.raises(ValueError, match="model_ids"):
            DemandProfiler(
                sample_demand_matrix,
                response_matrix=sample_response_matrix,
                model_ids=["a", "b"],  # only 2, but 5 models
            )


class TestDemandProfile:
    def test_demand_profile(self, sample_demand_matrix):
        profiler = DemandProfiler(sample_demand_matrix)
        profile = profiler.demand_profile()

        assert isinstance(profile, DemandProfile)
        assert len(profile.dimensions) == 5
        assert len(profile.mean_demands) == 5
        assert all(profile.mean_demands >= 0)

    def test_summary(self, sample_demand_matrix):
        profiler = DemandProfiler(sample_demand_matrix)
        profile = profiler.demand_profile()
        summary = profile.summary()

        assert "mean" in summary.columns
        assert "std" in summary.columns
        assert "nonzero_frac" in summary.columns
        assert len(summary) == 5

    def test_level_distributions(self, sample_demand_matrix):
        profiler = DemandProfiler(sample_demand_matrix)
        profile = profiler.demand_profile()

        assert "logical_reasoning" in profile.level_distributions
        assert isinstance(profile.level_distributions["logical_reasoning"], pd.Series)


class TestBenchmarkValidity:
    def test_sensitivity_specificity(self, sample_demand_matrix):
        profiler = DemandProfiler(sample_demand_matrix)
        validity = profiler.benchmark_validity(
            benchmark_name="TestBench",
            target_dimensions=["logical_reasoning", "quantitative_reasoning"],
        )

        assert isinstance(validity, BenchmarkValidity)
        assert validity.benchmark_name == "TestBench"
        assert len(validity.sensitivity) == 2  # 2 target dimensions
        assert len(validity.specificity) == 3  # 3 non-target dimensions
        assert 0 <= validity.overall_sensitivity <= 1
        assert 0 <= validity.overall_specificity <= 1

    def test_missing_target_dimension(self, sample_demand_matrix):
        profiler = DemandProfiler(sample_demand_matrix)
        validity = profiler.benchmark_validity(
            benchmark_name="TestBench",
            target_dimensions=["nonexistent_dimension"],
        )
        assert validity.sensitivity.iloc[0]["sensitivity"] == 0.0

    def test_summary(self, sample_demand_matrix):
        profiler = DemandProfiler(sample_demand_matrix)
        validity = profiler.benchmark_validity(
            benchmark_name="TestBench",
            target_dimensions=["logical_reasoning"],
        )
        summary = validity.summary()
        assert "type" in summary.columns
        assert set(summary["type"].unique()) <= {"target", "extraneous"}


class TestAbilityProfile:
    def test_ability_estimation(self, sample_demand_matrix, sample_response_matrix, sample_model_ids):
        profiler = DemandProfiler(
            sample_demand_matrix,
            response_matrix=sample_response_matrix,
            model_ids=sample_model_ids,
        )
        abilities = profiler.ability_profile()

        assert isinstance(abilities, pd.DataFrame)
        assert list(abilities.index) == sample_model_ids
        assert len(abilities.columns) == 5

    def test_no_response_matrix(self, sample_demand_matrix):
        profiler = DemandProfiler(sample_demand_matrix)
        with pytest.raises(ValueError, match="response_matrix"):
            profiler.ability_profile()


class TestPredictPerformance:
    def test_prediction_shape(self, sample_demand_matrix, sample_response_matrix, sample_model_ids):
        profiler = DemandProfiler(
            sample_demand_matrix,
            response_matrix=sample_response_matrix,
            model_ids=sample_model_ids,
        )
        predictions = profiler.predict_performance()

        assert predictions.shape == (5, 20)
        assert all(predictions.values.flatten() >= 0)
        assert all(predictions.values.flatten() <= 1)

    def test_prediction_with_custom_abilities(self, sample_demand_matrix, sample_response_matrix, sample_model_ids):
        profiler = DemandProfiler(
            sample_demand_matrix,
            response_matrix=sample_response_matrix,
            model_ids=sample_model_ids,
        )
        # Custom ability profile
        abilities = pd.DataFrame(
            np.ones((5, 5)) * 3.0,
            index=sample_model_ids,
            columns=sample_demand_matrix.columns,
        )
        predictions = profiler.predict_performance(ability_profiles=abilities)
        assert predictions.shape == (5, 20)


class TestCharacteristicCurves:
    def test_curves(self, sample_demand_matrix, sample_response_matrix, sample_model_ids):
        profiler = DemandProfiler(
            sample_demand_matrix,
            response_matrix=sample_response_matrix,
            model_ids=sample_model_ids,
        )
        curves = profiler.characteristic_curves(n_bins=4)

        assert isinstance(curves, dict)
        assert len(curves) > 0
        for dim, df in curves.items():
            assert "demand_level" in df.columns
            assert all(mid in df.columns for mid in sample_model_ids)


class TestDimensionCorrelations:
    def test_correlations(self, sample_demand_matrix):
        profiler = DemandProfiler(sample_demand_matrix)
        corr = profiler.dimension_correlations()

        assert corr.shape == (5, 5)
        # Diagonal should be 1
        for dim in sample_demand_matrix.columns:
            assert corr.loc[dim, dim] == pytest.approx(1.0)


class TestPlots:
    def test_plot_demand_profile(self, sample_demand_matrix):
        profiler = DemandProfiler(sample_demand_matrix)
        profile = profiler.demand_profile()
        fig = plot_demand_profile(profile, title="Test")
        assert fig is not None
        plt.close(fig)

    def test_plot_ability_vs_demand(self, sample_demand_matrix, sample_response_matrix, sample_model_ids):
        profiler = DemandProfiler(
            sample_demand_matrix,
            response_matrix=sample_response_matrix,
            model_ids=sample_model_ids,
        )
        profile = profiler.demand_profile()
        abilities = profiler.ability_profile()
        fig = plot_ability_vs_demand(abilities, profile, model_ids=["GPT-4", "Claude-3"])
        assert fig is not None
        plt.close(fig)

    def test_plot_sensitivity_specificity(self, sample_demand_matrix):
        profiler = DemandProfiler(sample_demand_matrix)
        validity = profiler.benchmark_validity(
            benchmark_name="TestBench",
            target_dimensions=["logical_reasoning"],
        )
        fig = plot_sensitivity_specificity(validity)
        assert fig is not None
        plt.close(fig)


# Need to import plt for closing figures in tests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
