from typing import Any, Tuple
import warnings

import numpy as np
import scipy.spatial.distance
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import eigs

from rankcentrality.types import Scores, Comparisons, ComparisonResults, Matrix


def get_transition_matrix(
    n_items: int, comps: Comparisons, comp_results: ComparisonResults
) -> Matrix:
    """Computes the Markov transition matrix resulting for a set of comparisons.

    Args:
        n_items: The number of items being compared.
        comps: Pairs of items that were compared.
        comp_results: Results of pairwise comparisons passed in `comps`.
    Returns:
        The transition matrix Q
    """
    Q = np.zeros((n_items, n_items))
    for (i, j), y in zip(comps, comp_results):
        Q[[i, j][1 - y], [i, j][y]] += 1
    Q = np.divide(Q, len(comps))
    row_sums = Q.sum(axis=1)
    Q[np.diag_indices(n_items)] += 1 - row_sums
    return Q


def is_strongly_connected(A: Matrix) -> bool:
    """Determines whether a graph is strongly connected
  
    Args:
        A: n x n adjacency matrix
    Returns:
        True if adjacency matrix is strongly connected.
    """

    def _dfs(node, adj, visited):
        visited[node] = True
        for nextnode in adj[node].nonzero()[1]:
            if not visited[nextnode]:
                _dfs(nextnode, adj, visited)

    n_items = A.shape[0]
    A = coo_matrix(A)

    visited = np.zeros(n_items, dtype=bool)
    _dfs(0, csr_matrix(A), visited)
    if not np.all(visited):
        return False

    visited[:] = 0
    _dfs(0, csr_matrix(A.T), visited)
    return np.all(visited)


def get_stationary_distribution(Q: Matrix, skip_check: bool = False) -> Scores:
    """Gets the stationary distribution of a Markov transition matrix.

    Args:
        Q: The transition matrix.
        skip_check: If set to True, this function will not check if Q is an
            ergodic matrix, which can speed up the computation but may lead
            to weird results if the matrix is not ergodic (for example, the
            resulting vector might have negative entries).
    """
    n_items = Q.shape[0]
    if not skip_check and not is_strongly_connected(Q):
        # Markov chain is not irreducible don't bother finding stationary
        # distribution.
        return np.ones(n_items) / n_items
    evals, evecs = eigs(Q.T, 1)
    if not np.allclose(evals[0], 1):  # eigenvalue should be close to 1
        warnings.warn(f"Got an eigenvalue of {evals[0]} when expecting eigenvalue 1")
    if not np.allclose(evecs.imag, 0):  # eigenvector should be real
        warnings.warn("Got complex eigenvector")

    evecs = evecs.real
    evecs = evecs / evecs.sum()
    assert np.all(
        evecs > -np.finfo(evecs.dtype).eps
    ), "Eigenvector (stationary distribution) has negative values."
    return np.abs(evecs.flatten())


def d_lam(n_items: int, lam: float = np.exp(-3)) -> Matrix:
    """Returns the regularizer matrix D_lambda.

    Args:
        n_items: The number of items (and hence the output matrix is of shape
            (n_items, n_items))
        lam: The lambda parameter. The non-diagonal entries of the output
            will be lam/n_items.
    """
    D = (1 - lam) * np.eye(n_items)
    D += (lam / n_items) * np.ones((n_items, n_items))
    return D


def get_affinity_matrix(
    X: Matrix,
    kernel_width: float,
    threshold: float = 0,
    metric: str = "euclidean",
    stochastic: bool = True,
) -> Matrix:
    """Returns the affinity matrix computed using feature vectors.

    Args:
        X: A matrix of shape (n_items, n_dimensions)
        kernel_width: The kernel width sigma used to compute affinity.
        metric: A distance metric from which to compute affinity. Any value
            supported by scipy's pairwise distance methods
            (scipy.spatial.distance.pdist) is acceptable.
        stochastic: (optional) Whether the resulting matrix should be
            normalized to be row stochastic (i.e., each row sums to 1.). If
            set to False, it is the user's responsibility to normalize rows
            before running RankCentrality. Defaults to True.
    """
    A = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X, metric))
    A = np.exp(-(A ** 2) / kernel_width ** 2) - threshold
    A[A < 0] = 0
    if stochastic:
        A = A / A.sum(axis=1)[:, None]
    return A
