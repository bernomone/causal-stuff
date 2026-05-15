"""Covariate whitening utilities for Mahalanobis distance matching in DoWhy."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg


def whiten_covariates(
    data: pd.DataFrame,
    covariate_cols: list[str],
    epsilon: float = 1e-8,
) -> pd.DataFrame:
    """
    Apply Mahalanobis whitening transform to covariates.

    After whitening, Euclidean distance in the transformed space equals
    Mahalanobis distance in the original space. This enables using DoWhy's
    standard distance matching with Mahalanobis-equivalent distances.

    Mathematical relationship:
    - Mahalanobis distance: d_M(x, y) = sqrt((x - y)^T Σ^(-1) (x - y))
    - Transform: x' = (x - μ) @ Σ^(-1/2)
    - Euclidean on x': d_E(x', y') = d_M(x, y)

    The whitening transform standardizes each covariate to zero mean and unit
    variance, AND decorrelates them (covariance matrix becomes identity).

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset containing covariates to whiten
    covariate_cols : list of str
        Names of covariate columns to apply whitening transform
    epsilon : float, default=1e-8
        Small value added to eigenvalues to avoid division by zero
        for near-singular covariance matrices

    Returns
    -------
    pd.DataFrame
        Copy of data with whitened covariates (original values replaced)

    Examples
    --------
    >>> data_whitened = whiten_covariates(data, ['age', 'educ', 're74'])
    >>> # Now Euclidean distance in data_whitened = Mahalanobis in data
    >>> model = CausalModel(data=data_whitened, ...)
    >>> # DoWhy's distance_matching will use Mahalanobis-equivalent distances

    Notes
    -----
    After whitening:
    - Mean of each covariate = 0
    - Variance of each covariate = 1
    - Covariance between any two covariates = 0 (identity covariance matrix)

    This transform is useful when:
    1. Covariates have very different scales (e.g., age vs income)
    2. Covariates are correlated
    3. You want Mahalanobis distance but the estimator only supports Euclidean

    See Also
    --------
    sklearn.preprocessing.StandardScaler : Only standardizes, doesn't decorrelate
    scipy.cluster.vq.whiten : Similar but uses stddev instead of full covariance
    """
    data = data.copy()
    X = data[covariate_cols].values

    # Center the data (zero mean)
    mean = X.mean(axis=0)
    X_centered = X - mean

    # Compute covariance matrix
    cov = np.cov(X_centered.T)

    # Compute inverse square root of covariance (whitening matrix)
    # Using eigendecomposition: Σ^(-1/2) = Q @ Λ^(-1/2) @ Q^T
    # where Q are eigenvectors and Λ are eigenvalues
    eigenvalues, eigenvectors = linalg.eigh(cov)

    # Protect against near-zero eigenvalues (near-singular covariance)
    eigenvalues = np.maximum(eigenvalues, epsilon)

    # Compute Σ^(-1/2)
    inv_sqrt_eigenvalues = 1.0 / np.sqrt(eigenvalues)
    whitening_matrix = eigenvectors @ np.diag(inv_sqrt_eigenvalues) @ eigenvectors.T

    # Apply whitening transform
    X_whitened = X_centered @ whitening_matrix

    # Replace original covariates with whitened versions
    data[covariate_cols] = X_whitened

    return data


def verify_whitening(data: pd.DataFrame, covariate_cols: list[str]) -> dict[str, float]:
    """
    Verify that covariates have been properly whitened.

    Checks that the covariance matrix is approximately the identity matrix
    (diagonal elements ≈ 1, off-diagonal elements ≈ 0).

    Parameters
    ----------
    data : pd.DataFrame
        Data that should contain whitened covariates
    covariate_cols : list of str
        Names of covariate columns to check

    Returns
    -------
    dict
        Statistics about the whitening quality:
        - 'is_identity': bool, True if covariance ≈ identity (tolerance 1e-5)
        - 'max_deviation': float, largest deviation from identity matrix
        - 'mean_abs_off_diagonal': float, average absolute off-diagonal value

    Examples
    --------
    >>> data_w = whiten_covariates(data, ['age', 'educ'])
    >>> stats = verify_whitening(data_w, ['age', 'educ'])
    >>> print(f"Properly whitened: {stats['is_identity']}")
    """
    X = data[covariate_cols].values
    cov = np.cov(X.T)
    identity = np.eye(len(covariate_cols))

    deviation = np.abs(cov - identity)
    max_dev = deviation.max()

    # Get off-diagonal elements
    mask = ~np.eye(len(covariate_cols), dtype=bool)
    off_diag = np.abs(cov[mask])
    mean_off_diag = off_diag.mean() if len(off_diag) > 0 else 0.0

    return {
        "is_identity": np.allclose(cov, identity, atol=1e-5),
        "max_deviation": float(max_dev),
        "mean_abs_off_diagonal": float(mean_off_diag),
    }
