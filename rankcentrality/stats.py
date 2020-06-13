from typing import List, Tuple, Optional

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import numba
import pandas as pd
from scipy.stats import kendalltau

from rankcentrality.types import (
    Scores,
    Matrix,
    ExperimentResults,
    ExperimentMetrics,
    Metrics,
)

_line_styles = ["-", ":", "-.", "--"]

canonical_loss_name = {
    "kendalltau": "kendalltau",
    "ell_2": "ell_2",
    "exp_test_error": "exp_test_error",
    # aliases
    "kt": "kendalltau",
    "2": "ell_2",
    "l2": "ell_2",
    "ete": "exp_test_error",
}


@numba.njit
def _getP(scores: Scores) -> Matrix:
    """Returns the BTL preference matrix for the scores given."""
    n = len(scores)
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            P[i, j] = scores[j] / (scores[i] + scores[j])
    return P


@numba.njit
def expected_test_error(scores_hat: Scores, scores: Scores) -> float:
    """Computes the expected test error.

    Args:
        scores: the vector of true scores.
        scores_hat: a vector of empirically estimated scores.
    Returns:
        The probability that scores_hat will correctly predict the outcome of
        a pairwise comparison between a pair of items chosen uniformly at
        random from all possible pairs.
    """
    scores = scores.flatten()
    scores_hat = scores_hat.flatten()
    n_items = len(scores)
    P = _getP(scores)
    numerator = 0
    denominator = 0
    for i in range(n_items):
        for j in range(n_items):
            if i == j:
                continue
            pred = float(int(scores_hat[j] > scores_hat[i]))
            numerator += np.abs(pred - P[i, j])
            denominator += 1
    assert denominator == (n_items * (n_items - 1)), "Internal error."
    return numerator / denominator


def compute_loss(w_hat: Scores, w: Scores, loss: str) -> float:
    r"""Computes the loss of an empirical score vector.

    Supported loss functions:
      - The "kendalltau" (alias "kt") loss is Kendall's Tau-b statistic.
      - The "ell_2" (alias "l2", "2") loss is the $\ell_2$ loss. Note that
        per convention, w and w_hat are canonicalized to be of ell_1 norm 1
        (i.e., their entries sum to 1). Further, the returned loss is
        relative to the norm of w. That is, the returned loss is
        || w_hat - w || / || w ||,
        where || . || is the ell_2 norm.
      - The expected test error (alias "ete") loss is the expected test error
        assuming the BTL model. That is, if we draw a pair of items uniformly
        at random, the expected test error is probability that the empirical
        scores and true scores agree on which item is more likely to be
        preferred.

    Args:
        w_hat: The empirical score vector.
        w: The true score vector.
        loss: A loss function as described above.

    Returns:
        The computed loss.
    """
    assert isinstance(w_hat, Scores)
    assert isinstance(w, Scores)
    try:
        loss = canonical_loss_name[loss]
    except KeyError:
        raise ValueError(f"Did not recognize loss function {loss}.")
    if loss == "kendalltau":
        return kendalltau(w, w_hat, method="asymptotic")[0]
    elif loss == "ell_2":
        # Flatten the arrays. If a non-flattened version slips past the checks
        # above, numpy might perform an undesirable broadcast.
        w, w_hat = w.flatten(), w_hat.flatten()
        w = w / np.linalg.norm(w, 1)
        w_hat = w_hat / np.linalg.norm(w_hat, 1)
        return np.linalg.norm(w - w_hat, 2) / np.linalg.norm(w, 2)
    elif loss == "exp_test_error":
        return expected_test_error(w_hat, w)


def compute_experiment_metrics(
    results: ExperimentResults,
    w: Scores,
    loss: str,
    num_comps_length: int,
    num_repetitions: int,
) -> ExperimentMetrics:
    """Computes metrics for an experiment.

    An experiment is assumed to have multiple algorithms, emprical results
    for different number of pairwise comparisons, and, for each such
    scenario, a number of repetitions (with fresh sets of comparisons for
    each repetition).

    This function computes the mean and standard error of the loss of each
    algorithm for each number of comparisons.
    """
    experiment_metrics = {}
    for algo, algo_results in results.items():
        # algo_results[num_comps_index][repetition_index] is a score vector of
        # the corresponding experiment
        losses = np.array(
            [
                [compute_loss(w_hat, w, loss) for w_hat in repetitions]
                for repetitions in algo_results
            ]
        )
        assert losses.shape == (num_comps_length, num_repetitions,), (
            f"Not all algorithms had {num_comps_length} different values "
            f"of number of comparisons and {num_repetitions} repetitions."
        )
        experiment_metrics[algo] = Metrics(
            means=np.mean(losses, axis=1),
            std_errs=np.std(losses, axis=1) / np.sqrt(num_repetitions),
        )
    return experiment_metrics


def plot_experiment_metrics(
    experiment_metrics: ExperimentMetrics,
    num_comps_list: List[int],
    loss: str,
    num_comps_label: Optional[str] = None,
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Plots experiments metrics as a Matplotlib figure."""
    loss = canonical_loss_name[loss]
    f, ax = plt.subplots()
    ax.set_xlabel(num_comps_label or "Number of comparisons")
    ax.set_ylabel(
        {
            "kendalltau": "Kendall Tau metric",
            "ell_2": r"$\|\hat\pi - \pi\|/\|\pi\|$",
            "exp_test_error": "Expected Test Error",
        }[loss]
    )
    ax.set_xscale("log")
    if loss == "ell_2":
        ax.set_yscale("log")
    for i, algo in enumerate(experiment_metrics.keys()):
        ax.errorbar(
            num_comps_list,
            experiment_metrics[algo].means,
            yerr=experiment_metrics[algo].std_errs,
            label=algo,
            linestyle=_line_styles[i % 4],
        )
    ax.legend()
    return (f, ax)


def experiment_metrics_to_dataframe(
    experiment_metrics: ExperimentMetrics, num_comps_list: List[int]
) -> pd.DataFrame:
    """Returns the experiment metrics as a Pandas DataFrame."""
    columns = {"num_comps": num_comps_list}
    for algo, metrics in experiment_metrics.items():
        columns[algo] = metrics.means
        columns[algo + "_std_err"] = metrics.std_errs
    df = pd.DataFrame(columns).set_index("num_comps")
    return df
