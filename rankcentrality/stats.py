from typing import List, Tuple

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from rankcentrality.types import Scores, ExperimentResults, ExperimentMetrics, Metrics

_line_styles = ["-", ":", "-.", "--"]

canonical_loss_name = {
    "kendalltau": "kendalltau",
    "ell_2": "ell_2",
    # aliases
    "kt": "kendalltau",
    "2": "ell_2",
    "l2": "ell_2",
}


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
    experiment_metrics: ExperimentMetrics, num_comps_list: List[int], loss: str
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Plots experiments metrics as a Matplotlib figure."""
    loss = canonical_loss_name[loss]
    f, ax = plt.subplots()
    ax.set_xlabel("Number of comparisons")
    ax.set_ylabel(
        {"kendalltau": "Kendall Tau metric", "ell_2": r"$\|\hat\pi - \pi\|/\|\pi\|$",}[
            loss
        ]
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
