"""Preprocessing helpers for whitening and conditioning checks."""

from typing import Optional

import jax
import jax.numpy as jnp
from sklearn.base import BaseEstimator, TransformerMixin


class Whitener(BaseEstimator, TransformerMixin):
    """A Scikit-learn sytle transformer for matrix whitening."""

    def __init__(self, eps=1e-10):
        """
        Args:
            eps: Small value added to eigenvalues to avoid division by zero.

        Returns:
            An instance of `Whitener` transformer.
        """
        self.eps = eps

    def fit(self, X: jax.Array, y: Optional[jax.Array] = None):
        """
        Fit the whitening transformer to the data.

        Args:
            X : Input data of shape (n_samples, n_features).
            y (ignored): Not used, for API compatibility.

        Returns:
            self: Fitted transformer.
        """
        self.mu = jnp.mean(X, axis=0)
        cov = jnp.cov(X - self.mu, rowvar=False)
        eigenvalues, eigenvectors = jnp.linalg.eigh(cov)
        self.W = (
            eigenvectors
            @ jnp.diag(1.0 / jnp.sqrt(eigenvalues + self.eps))
            @ eigenvectors.T
        )
        return self

    def transform(self, X: jax.Array):
        """Apply the whitening transformation to the data.

        Args:
            X: Input data of shape (n_samples, n_features).

        Returns:
            Whitened data.
        """
        return (X - self.mu) @ self.W


def condition_number(X: jax.Array):
    """Calculates the condition number of a dataset's covariance matrix.

    Args:
        X: Dataset of shape [M, N].

    Returns:
        The condition number of the covariance matrix between the columns
        of `X`.
    """
    cov = jnp.cov(X, rowvar=False)
    singular_values = jnp.linalg.svd(cov, compute_uv=False)
    return singular_values.max() / singular_values.min()
