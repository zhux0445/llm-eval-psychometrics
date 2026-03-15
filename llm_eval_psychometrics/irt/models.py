"""IRT model fitting, wrapping the girth library.

Supports 2PL and 3PL Item Response Theory models for analyzing
LLM benchmark response matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import girth
import numpy as np
import pandas as pd


@dataclass
class IRTResults:
    """Container for IRT model fitting results.

    Attributes:
        item_params: DataFrame with columns 'discrimination', 'difficulty',
            and optionally 'guessing' (3PL). Indexed by item id.
        model_ability: Series of theta (ability) estimates, indexed by model id.
        model_type: The IRT model used ('2PL' or '3PL').
        aic: Akaike Information Criterion dict (2PL only, None for 3PL).
        bic: Bayesian Information Criterion dict (2PL only, None for 3PL).
    """

    item_params: pd.DataFrame
    model_ability: pd.Series
    model_type: str
    aic: dict | None = None
    bic: dict | None = None
    _response_matrix: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)

    def flag_poor_items(
        self,
        min_discrimination: float = 0.3,
        max_difficulty_abs: float = 3.0,
        max_guessing: float = 0.4,
    ) -> pd.DataFrame:
        """Flag items with poor psychometric properties.

        An item is flagged if ANY of the following hold:
        - Discrimination < min_discrimination (item doesn't differentiate ability levels)
        - |Difficulty| > max_difficulty_abs (item is too easy or too hard)
        - Guessing > max_guessing (3PL only; high guessing undermines measurement)

        Args:
            min_discrimination: Minimum acceptable discrimination. Default 0.3.
            max_difficulty_abs: Maximum acceptable |difficulty|. Default 3.0.
            max_guessing: Maximum acceptable guessing parameter. Default 0.4.

        Returns:
            DataFrame of flagged items with a 'reasons' column listing why each
            item was flagged.
        """
        params = self.item_params
        reasons: dict[str, list[str]] = {}

        for idx, row in params.iterrows():
            item_reasons = []
            if row["discrimination"] < min_discrimination:
                item_reasons.append(
                    f"low discrimination ({row['discrimination']:.3f} < {min_discrimination})"
                )
            if abs(row["difficulty"]) > max_difficulty_abs:
                item_reasons.append(
                    f"extreme difficulty ({row['difficulty']:.3f}, |b| > {max_difficulty_abs})"
                )
            if "guessing" in row.index and row["guessing"] > max_guessing:
                item_reasons.append(
                    f"high guessing ({row['guessing']:.3f} > {max_guessing})"
                )
            if item_reasons:
                reasons[idx] = item_reasons

        if not reasons:
            return pd.DataFrame(columns=["reasons"])

        result = params.loc[list(reasons.keys())].copy()
        result["reasons"] = [reasons[idx] for idx in result.index]
        return result

    def item_information(
        self, theta: np.ndarray | None = None
    ) -> pd.DataFrame:
        """Compute item information for each item across theta values.

        Item information quantifies how precisely an item measures ability at
        each theta level. For the 2PL model:
            I(θ) = a² * P(θ) * (1 - P(θ))
        where a = discrimination and P(θ) is the probability of correct response.

        Args:
            theta: Array of ability values. Defaults to linspace(-4, 4, 81).

        Returns:
            DataFrame with theta as index and one column per item.
        """
        if theta is None:
            theta = np.linspace(-4, 4, 81)

        a = self.item_params["discrimination"].values
        b = self.item_params["difficulty"].values
        c = (
            self.item_params["guessing"].values
            if "guessing" in self.item_params.columns
            else np.zeros_like(a)
        )

        # theta: (T,), a/b/c: (I,) -> broadcast to (T, I)
        theta_2d = theta[:, np.newaxis]
        # 3PL ICC: P(θ) = c + (1-c) / (1 + exp(-a(θ-b)))
        p = c + (1 - c) / (1 + np.exp(-a * (theta_2d - b)))
        # Fisher information for 3PL:
        #   I(θ) = a² * (P - c)² * (1 - P) / ((1 - c)² * P)
        # For 2PL (c=0) this reduces to: I(θ) = a² * P * (1 - P)
        numerator = a**2 * (p - c) ** 2 * (1 - p)
        denominator = (1 - c) ** 2 * p
        with np.errstate(divide="ignore", invalid="ignore"):
            info = np.where(denominator > 1e-10, numerator / denominator, 0.0)

        return pd.DataFrame(
            info, index=theta, columns=self.item_params.index
        )

    def test_information(self, theta: np.ndarray | None = None) -> pd.Series:
        """Compute total test information curve (sum of all item information).

        Test information is the sum of individual item information functions.
        Higher information means more precise ability estimation at that theta.

        Args:
            theta: Array of ability values. Defaults to linspace(-4, 4, 81).

        Returns:
            Series with theta as index and total information as values.
        """
        item_info = self.item_information(theta)
        return item_info.sum(axis=1).rename("test_information")


class IRTAnalyzer:
    """Fit IRT models to LLM benchmark response matrices.

    Uses the girth library for parameter estimation via Marginal Maximum
    Likelihood (MML).

    Args:
        model: IRT model type. '2PL' (two-parameter logistic) or
            '3PL' (three-parameter logistic). Default '2PL'.

    Example:
        >>> import numpy as np
        >>> from llm_eval_psychometrics.irt import IRTAnalyzer
        >>> analyzer = IRTAnalyzer(model="2PL")
        >>> response_matrix = np.random.randint(0, 2, size=(10, 50))
        >>> results = analyzer.fit(response_matrix)
        >>> results.item_params.head()
    """

    SUPPORTED_MODELS = ("2PL", "3PL")

    def __init__(self, model: Literal["2PL", "3PL"] = "2PL") -> None:
        model = model.upper()
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model '{model}'. Choose from {self.SUPPORTED_MODELS}."
            )
        self.model = model

    def fit(
        self,
        response_matrix: np.ndarray,
        item_ids: list[str] | None = None,
        model_ids: list[str] | None = None,
    ) -> IRTResults:
        """Fit the IRT model to a binary response matrix.

        Args:
            response_matrix: Binary matrix of shape (n_models, n_items) where
                1 = correct and 0 = incorrect. Each row is one LLM, each
                column is one benchmark item.
            item_ids: Optional names for items. Defaults to ['item_0', ...].
            model_ids: Optional names for models. Defaults to ['model_0', ...].

        Returns:
            IRTResults with estimated item parameters and model abilities.

        Raises:
            ValueError: If the response matrix has invalid shape or values.
        """
        response_matrix = np.asarray(response_matrix, dtype=float)
        self._validate_input(response_matrix)
        # girth internally uses arrays as indices, which requires int dtype
        response_matrix_int = response_matrix.astype(int)

        n_models, n_items = response_matrix.shape

        if item_ids is None:
            item_ids = [f"item_{i}" for i in range(n_items)]
        if model_ids is None:
            model_ids = [f"model_{i}" for i in range(n_models)]

        if len(item_ids) != n_items:
            raise ValueError(
                f"item_ids length ({len(item_ids)}) != number of items ({n_items})."
            )
        if len(model_ids) != n_models:
            raise ValueError(
                f"model_ids length ({len(model_ids)}) != number of models ({n_models})."
            )

        # girth expects [items x participants], so transpose
        data = response_matrix_int.T

        if self.model == "2PL":
            result = girth.twopl_mml(data)
            discrimination = result["Discrimination"]
            difficulty = result["Difficulty"]
            ability = result["Ability"]
            aic = result.get("AIC")
            bic = result.get("BIC")

            item_params = pd.DataFrame(
                {"discrimination": discrimination, "difficulty": difficulty},
                index=item_ids,
            )

        else:  # 3PL
            result = girth.threepl_mml(data)
            discrimination = result["Discrimination"]
            difficulty = result["Difficulty"]
            guessing = result["Guessing"]
            # 3PL doesn't return ability directly; estimate via EAP
            ability = girth.ability_eap(data, difficulty, discrimination)
            aic = None
            bic = None

            item_params = pd.DataFrame(
                {
                    "discrimination": discrimination,
                    "difficulty": difficulty,
                    "guessing": guessing,
                },
                index=item_ids,
            )

        item_params.index.name = "item_id"
        model_ability = pd.Series(ability, index=model_ids, name="ability")
        model_ability.index.name = "model_id"

        return IRTResults(
            item_params=item_params,
            model_ability=model_ability,
            model_type=self.model,
            aic=aic,
            bic=bic,
            _response_matrix=response_matrix,
        )

    @staticmethod
    def _validate_input(response_matrix: np.ndarray) -> None:
        """Validate the response matrix format.

        Args:
            response_matrix: The matrix to validate.

        Raises:
            ValueError: If matrix is not 2D, not binary, or too small.
        """
        if response_matrix.ndim != 2:
            raise ValueError(
                f"response_matrix must be 2D, got shape {response_matrix.shape}. "
                "Expected (n_models, n_items)."
            )

        n_models, n_items = response_matrix.shape
        if n_models < 2:
            raise ValueError(
                f"Need at least 2 models (rows), got {n_models}. "
                "IRT requires variance across respondents."
            )
        if n_items < 2:
            raise ValueError(
                f"Need at least 2 items (columns), got {n_items}."
            )

        unique_vals = np.unique(response_matrix[~np.isnan(response_matrix)])
        if not np.all(np.isin(unique_vals, [0, 1])):
            raise ValueError(
                f"response_matrix must contain only 0, 1, or NaN. "
                f"Found values: {unique_vals}. "
                "For polytomous data, consider graded response models."
            )
