import numpy as np
import pytest

import rankcentrality as rc


@pytest.mark.parametrize("loss", ["kt", "2", "ete"])
def test_generates_metrics(loss):
    # num_items = 2
    num_comps = [5, 10, 20]
    num_repetitions = 5
    w = np.array([0.2, 0.8])
    dummy_results = {
        "algo1": [[np.array([0.2, 0.8]),] * num_repetitions] * len(num_comps),
        "algo2": [[np.array([0.8, 0.2]),] * num_repetitions] * len(num_comps),
    }
    metrics = rc.stats.compute_experiment_metrics(
        dummy_results,
        w,
        loss=loss,
        num_comps_length=len(num_comps),
        num_repetitions=num_repetitions,
    )
    expected_mean = {"kt": 1, "2": 0, "ete": 0.2}
    assert np.allclose(metrics["algo1"].means, expected_mean[loss])
    assert np.allclose(metrics["algo1"].std_errs, 0)
    assert not np.allclose(metrics["algo2"].means, expected_mean[loss])
    assert np.allclose(metrics["algo1"].std_errs, 0)


def test_generates_plots():
    dummy_metrics = {
        "algo1": rc.types.Metrics(
            np.array([0.2, 0.3, 0.4]), np.array([0.01, 0.01, 0.01])
        ),
        "algo2": rc.types.Metrics(
            np.array([0.1, 0.2, 0.3]), np.array([0.01, 0.01, 0.01])
        ),
    }
    f, ax = rc.stats.plot_experiment_metrics(
        dummy_metrics, num_comps_list=[5, 10, 20], loss="2"
    )


def test_generates_dataframe():
    dummy_metrics = {
        "algo1": rc.types.Metrics(
            np.array([0.2, 0.3, 0.4]), np.array([0.01, 0.01, 0.01])
        ),
        "algo2": rc.types.Metrics(
            np.array([0.1, 0.2, 0.3]), np.array([0.01, 0.01, 0.01])
        ),
    }
    df = rc.stats.experiment_metrics_to_dataframe(
        dummy_metrics, num_comps_list=[5, 10, 20]
    )
