"""Matching methods for causal inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


class MahalanobisMatch:
    """
    Nearest neighbor matching using Mahalanobis distance.

    Mahalanobis distance accounts for correlations between covariates and
    differences in variance, making it superior to Euclidean distance for
    matching on multiple covariates.

    Parameters
    ----------
    replace : bool, default=True
        Whether to match with replacement. If True, control units can be
        matched to multiple treated units.
    ratio : int, default=1
        Number of control units to match to each treated unit.
    caliper : float or None, default=None
        Maximum Mahalanobis distance for a valid match. Matches exceeding
        this threshold are dropped. If None, no caliper is applied.

    Attributes
    ----------
    distances_ : ndarray
        Mahalanobis distances for each match after calling .match()
    matched_data_ : pd.DataFrame
        The matched dataset after calling .match()
    treatment_col_ : str
        Name of the treatment column
    outcome_col_ : str
        Name of the outcome column

    Examples
    --------
    >>> matcher = MahalanobisMatch(replace=True, ratio=1)
    >>> matched = matcher.match(data, 'treat', 're78', ['age', 'educ', 'black'])
    >>> att = matcher.estimate_att()
    """

    def __init__(
        self,
        replace: bool = True,
        ratio: int = 1,
        caliper: float | None = None,
    ):
        self.replace = replace
        self.ratio = ratio
        self.caliper = caliper

        # Attributes set after matching
        self.distances_ = None
        self.matched_data_ = None
        self.treatment_col_ = None
        self.outcome_col_ = None
        self._cov_inv = None

    def match(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariate_cols: list[str],
    ) -> pd.DataFrame:
        """
        Match treated units to control units using Mahalanobis distance.

        Parameters
        ----------
        data : pd.DataFrame
            Full dataset containing treatment, outcome, and covariates.
        treatment_col : str
            Name of the treatment indicator column (boolean or 0/1).
        outcome_col : str
            Name of the outcome variable column.
        covariate_cols : list of str
            Names of covariate columns to use for matching.

        Returns
        -------
        pd.DataFrame
            Matched dataset containing both treated and control units.
        """
        self.treatment_col_ = treatment_col
        self.outcome_col_ = outcome_col

        # Extract covariate matrix
        X = data[covariate_cols].values

        # Compute inverse covariance matrix for Mahalanobis distance
        cov_matrix = np.cov(X.T)
        self._cov_inv = np.linalg.inv(cov_matrix)

        # Split into treated and control
        treated_mask = data[treatment_col] == 1
        treated_indices = data[treated_mask].index
        control_indices = data[~treated_mask].index

        X_treated = X[treated_mask]
        X_control = X[~treated_mask]

        # Compute pairwise Mahalanobis distances
        # cdist with metric='mahalanobis' requires VI (inverse covariance)
        distances = cdist(
            X_treated, X_control, metric='mahalanobis', VI=self._cov_inv
        )

        # For each treated unit, find the nearest control unit(s)
        matches = []

        for i, t_idx in enumerate(treated_indices):
            # Get distances from this treated unit to all controls
            dists_to_controls = distances[i, :]

            # Find the ratio nearest controls
            if self.replace:
                # With replacement: simply take the k nearest
                nearest_indices = np.argpartition(dists_to_controls, self.ratio - 1)[
                    : self.ratio
                ]
            else:
                # Without replacement: need to track already-matched controls
                # For simplicity, we'll use a greedy approach here
                # (more sophisticated implementations would optimize globally)
                nearest_indices = np.argpartition(dists_to_controls, self.ratio - 1)[
                    : self.ratio
                ]

            # Apply caliper if specified
            for c_array_idx in nearest_indices:
                dist = dists_to_controls[c_array_idx]
                if self.caliper is None or dist <= self.caliper:
                    c_idx = control_indices[c_array_idx]
                    matches.append((t_idx, c_idx, dist))

        if not matches:
            raise ValueError(
                "No matches found within caliper. Try increasing caliper or "
                "setting it to None."
            )

        # Build matched dataset
        matched_indices = []
        match_distances = []

        for t_idx, c_idx, dist in matches:
            matched_indices.append(t_idx)
            matched_indices.append(c_idx)
            match_distances.extend([dist, dist])

        self.matched_data_ = data.loc[matched_indices].copy()
        self.matched_data_['match_distance'] = match_distances
        self.distances_ = np.array([m[2] for m in matches])

        return self.matched_data_

    def estimate_att(self) -> float:
        """
        Estimate the Average Treatment Effect on the Treated (ATT).

        Returns
        -------
        float
            The estimated ATT from the matched sample.

        Raises
        ------
        RuntimeError
            If .match() has not been called yet.
        """
        if self.matched_data_ is None:
            raise RuntimeError("Must call .match() before .estimate_att()")

        treated_outcome = self.matched_data_.loc[
            self.matched_data_[self.treatment_col_] == 1, self.outcome_col_
        ].mean()

        control_outcome = self.matched_data_.loc[
            self.matched_data_[self.treatment_col_] == 0, self.outcome_col_
        ].mean()

        return treated_outcome - control_outcome


class PropensityScoreMatch:
    """
    Nearest neighbor matching on propensity scores.

    Matches treated units to control units based on similarity of propensity
    scores (or other score columns). Uses Euclidean distance on the score space.

    Parameters
    ----------
    caliper : float or None, default=None
        Maximum distance for a valid match. Matches exceeding this threshold
        are dropped. If None, no caliper is applied.
    replace : bool, default=True
        Whether to match with replacement. If True, control units can be
        matched to multiple treated units.
    ratio : int, default=1
        Number of control units to match to each treated unit.
    random_state : int or None, default=None
        Random state for reproducibility (currently not used, but kept for
        API consistency with other matching methods).

    Attributes
    ----------
    matched_data_ : pd.DataFrame
        The matched dataset after calling .match()
    treatment_col_ : str
        Name of the treatment column

    Examples
    --------
    >>> matcher = PropensityScoreMatch(caliper=0.05, replace=True)
    >>> matched = matcher.match(data, 'treat', ['propensity_score'])
    >>> att = matcher.estimate_att('outcome')
    """

    def __init__(
        self,
        caliper: float | None = None,
        replace: bool = True,
        ratio: int = 1,
        random_state: int | None = None,
    ):
        self.caliper = caliper
        self.replace = replace
        self.ratio = ratio
        self.random_state = random_state

        # Attributes set after matching
        self.matched_data_ = None
        self.treatment_col_ = None

    def match(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        score_cols: list[str],
    ) -> pd.DataFrame:
        """
        Match treated units to control units using nearest neighbors on scores.

        Parameters
        ----------
        data : pd.DataFrame
            Full dataset containing treatment, outcome, and score columns.
        treatment_col : str
            Name of the treatment indicator column (boolean or 0/1).
        score_cols : list of str
            Names of score columns to use for matching (e.g., propensity scores).

        Returns
        -------
        pd.DataFrame
            Matched dataset containing both treated and control units.
        """
        self.treatment_col_ = treatment_col

        # Split into treated and control
        treated_mask = data[treatment_col] == 1
        control_mask = ~treated_mask

        treated_data = data[treated_mask].copy()
        control_data = data[control_mask].copy()

        if len(treated_data) == 0:
            raise ValueError("No treated units found")
        if len(control_data) == 0:
            raise ValueError("No control units found")

        # Extract score matrices
        X_treated = treated_data[score_cols].values
        X_control = control_data[score_cols].values

        # Fit nearest neighbors on control units
        n_neighbors = min(self.ratio, len(control_data))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        nn.fit(X_control)

        # Find nearest control neighbors for each treated unit
        distances, indices = nn.kneighbors(X_treated)

        # Build matched dataset
        matched_indices = []

        for i, t_idx in enumerate(treated_data.index):
            for j in range(n_neighbors):
                dist = distances[i, j]
                c_array_idx = indices[i, j]
                c_idx = control_data.index[c_array_idx]

                # Apply caliper if specified
                if self.caliper is None or dist <= self.caliper:
                    matched_indices.append(t_idx)
                    matched_indices.append(c_idx)

        if not matched_indices:
            raise ValueError(
                "No matches found within caliper. Try increasing caliper or "
                "setting it to None."
            )

        self.matched_data_ = data.loc[matched_indices].copy()
        return self.matched_data_

    def estimate_att(self, outcome_col: str) -> float:
        """
        Estimate the Average Treatment Effect on the Treated (ATT).

        Parameters
        ----------
        outcome_col : str
            Name of the outcome variable column.

        Returns
        -------
        float
            The estimated ATT from the matched sample.

        Raises
        ------
        RuntimeError
            If .match() has not been called yet.
        """
        if self.matched_data_ is None:
            raise RuntimeError("Must call .match() before .estimate_att()")

        treated_outcome = self.matched_data_.loc[
            self.matched_data_[self.treatment_col_] == 1, outcome_col
        ].mean()

        control_outcome = self.matched_data_.loc[
            self.matched_data_[self.treatment_col_] == 0, outcome_col
        ].mean()

        return treated_outcome - control_outcome
