"""Utilities for loading and generating benchmark data.

Provides synthetic data generators for testing and demos, plus helpers
for converting common benchmark formats to IRT-ready response matrices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_response_matrix(
    n_models: int = 20,
    n_items: int = 50,
    ability_range: tuple[float, float] = (-2.0, 2.0),
    difficulty_range: tuple[float, float] = (-2.0, 2.0),
    discrimination_range: tuple[float, float] = (0.5, 2.5),
    seed: int | None = 42,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Generate a synthetic IRT response matrix with known parameters.

    Simulates binary (correct/incorrect) responses using a 2PL IRT model,
    useful for testing and demos.

    Args:
        n_models: Number of LLMs (respondents).
        n_items: Number of benchmark items.
        ability_range: (min, max) for uniformly-spaced model abilities.
        difficulty_range: (min, max) for uniformly-spaced item difficulties.
        discrimination_range: (min, max) for uniform random discrimination.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (response_matrix, true_params) where:
        - response_matrix: (n_models, n_items) binary array
        - true_params: dict with keys 'ability', 'difficulty', 'discrimination'
    """
    rng = np.random.default_rng(seed)

    theta = np.linspace(ability_range[0], ability_range[1], n_models)
    b = np.linspace(difficulty_range[0], difficulty_range[1], n_items)
    a = rng.uniform(discrimination_range[0], discrimination_range[1], size=n_items)

    # 2PL ICC: P(θ) = 1 / (1 + exp(-a(θ - b)))
    prob = 1.0 / (1.0 + np.exp(-a[np.newaxis, :] * (theta[:, np.newaxis] - b[np.newaxis, :])))
    responses = (rng.random((n_models, n_items)) < prob).astype(int)

    true_params = {
        "ability": theta,
        "difficulty": b,
        "discrimination": a,
    }
    return responses, true_params


def simulate_mmlu_like(
    n_models: int = 15,
    subjects: list[str] | None = None,
    items_per_subject: int = 20,
    seed: int | None = 42,
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """Generate MMLU-like multi-subject benchmark data.

    Creates a response matrix spanning multiple subjects with varying
    difficulty distributions, mimicking the structure of MMLU.

    Args:
        n_models: Number of LLMs.
        subjects: Subject names. Defaults to a set of 5 common MMLU subjects.
        items_per_subject: Number of items per subject.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (response_matrix, model_ids, item_ids, item_subjects) where:
        - response_matrix: (n_models, total_items) binary array
        - model_ids: list of model name strings
        - item_ids: list of item id strings
        - item_subjects: list mapping each item to its subject
    """
    if subjects is None:
        subjects = [
            "abstract_algebra",
            "college_physics",
            "computer_science",
            "world_history",
            "clinical_knowledge",
        ]

    rng = np.random.default_rng(seed)

    model_ids = [
        f"model_{i}" for i in range(n_models)
    ]

    # Each model has an overall ability plus subject-specific noise
    base_ability = np.linspace(-1.5, 1.5, n_models)

    all_responses = []
    item_ids = []
    item_subjects = []

    # Different subjects have different difficulty profiles
    for subj_idx, subject in enumerate(subjects):
        # Shift difficulty per subject to create variety
        diff_center = rng.uniform(-1.0, 1.0)
        b = rng.normal(diff_center, 0.8, size=items_per_subject)
        a = rng.uniform(0.5, 2.0, size=items_per_subject)

        # Subject-specific ability modifier
        subj_ability = base_ability + rng.normal(0, 0.3, size=n_models)

        prob = 1.0 / (
            1.0 + np.exp(-a[np.newaxis, :] * (subj_ability[:, np.newaxis] - b[np.newaxis, :]))
        )
        resp = (rng.random((n_models, items_per_subject)) < prob).astype(int)

        all_responses.append(resp)
        for i in range(items_per_subject):
            item_ids.append(f"{subject}_{i}")
            item_subjects.append(subject)

    response_matrix = np.hstack(all_responses)
    return response_matrix, model_ids, item_ids, item_subjects
