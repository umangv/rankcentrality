from typing import NamedTuple

import pytest
import numpy as np
from flaky import flaky

import rankcentrality as rc
import rankcentrality.internal as internal
import rankcentrality.siamese as siamese


class PairwiseData(NamedTuple):
    n_items: int
    comps: rc.types.Comparisons
    comp_results: rc.types.ComparisonResults


# The following methods can be used to parameterize the various algorithms that
# we might test. They each use sane defaults (since we are often not making any
# assertions about the exact value of the results, these values don't matter).


# Algorithms that do *not* need item features:


def fit_btl(data: PairwiseData) -> rc.types.Scores:
    btl = rc.BTLMLE(data.n_items, data.comps, data.comp_results)
    return btl.run()


def fit_rc(data: PairwiseData) -> rc.types.Scores:
    rankc = rc.RankCentrality(data.n_items, data.comps, data.comp_results)
    return rankc.run()


def fit_rc_reg(data: PairwiseData) -> rc.types.Scores:
    rankc = rc.RankCentrality(data.n_items, data.comps, data.comp_results)
    return rankc.run_regularized(0.1)


def fit_rc_reg_decay(data: PairwiseData) -> rc.types.Scores:
    rankc = rc.RankCentrality(data.n_items, data.comps, data.comp_results)
    return rankc.run_regularized_decayed(0.1)


def fit_rc_custom(data: PairwiseData) -> rc.types.Scores:
    rankc = rc.RankCentrality(data.n_items, data.comps, data.comp_results)
    return rankc.run_custom_regularizer(internal.d_lam(data.n_items, 0.2))


# Algorithms that *do* need item features:


def fit_rc_diff(data: PairwiseData, features: rc.types.Matrix) -> rc.types.Scores:
    rankc = rc.RankCentrality(data.n_items, data.comps, data.comp_results)
    return rankc.run_diffused(features, 0.1, 0)


def fit_rc_diff_decayed(
    data: PairwiseData, features: rc.types.Matrix
) -> rc.types.Scores:
    rankc = rc.RankCentrality(data.n_items, data.comps, data.comp_results)
    return rankc.run_diffused_decayed(features, 0.1, 0)


def fit_ranksvm(data: PairwiseData, features: rc.types.Matrix) -> rc.types.Scores:
    ranksvm = rc.RankSVM(data.n_items, data.comps, data.comp_results, features)
    return ranksvm.run()


def fit_ranksvm_rf(data: PairwiseData, features: rc.types.Matrix) -> rc.types.Scores:
    ranksvm = rc.RankSVM(data.n_items, data.comps, data.comp_results, features)
    return ranksvm.run_random_features()


def fit_siamese(data: PairwiseData, features: rc.types.Matrix) -> rc.types.Scores:
    siam = siamese.SiameseNetRank(
        data.n_items,
        np.repeat(data.comps, 50, axis=0),
        np.repeat(data.comp_results, 50),
        features,
    )
    return siam.run(epochs=30)


# -----
# TESTS
# -----


@pytest.mark.parametrize(
    "algorithm", [fit_btl, fit_rc, fit_rc_reg, fit_rc_reg_decay, fit_rc_custom]
)
def test_no_features(algorithm):
    n_items = 3
    # For sake of having an ergodic markov chain, let items with lower score
    # beat items with higher scores one out of five times.
    comps = np.array([[0, 1], [0, 2], [1, 2]] * 5)
    comp_results = np.array([0, 0, 0] * 4 + [1, 1, 1])
    scores = algorithm(PairwiseData(n_items, comps, comp_results))
    assert scores[0] > scores[1]
    assert scores[1] > scores[2]


@pytest.mark.parametrize(
    "algorithm", [fit_btl, fit_rc, fit_rc_reg, fit_rc_reg_decay, fit_rc_custom]
)
def test_no_features_reorder(algorithm):
    """Ensure that tests don't pass because of coincidence in order."""
    n_items = 3
    # For sake of having an ergodic markov chain, let items with lower score
    # beat items with higher scores one out of five times.
    comps = np.array([[0, 1], [0, 2], [1, 2]] * 5)
    comp_results = np.array([0, 0, 1] * 4 + [1, 1, 0])
    scores = algorithm(PairwiseData(n_items, comps, comp_results))
    assert scores[0] > scores[2]
    assert scores[2] > scores[1]


# Note: drop fit_rc because it cannot handle non-ergodic markov chains
@pytest.mark.parametrize(
    "algorithm", [fit_btl, fit_rc_reg, fit_rc_reg_decay, fit_rc_custom]
)
def test_no_features_nonergodic(algorithm):
    n_items = 3
    comps = np.array([[0, 1], [1, 2]])
    comp_results = np.array([0, 0])
    scores = algorithm(PairwiseData(n_items, comps, comp_results))
    assert scores[0] > scores[1]
    assert scores[1] > scores[2]


# Note: only include fit_rc because it is the only one that needs to default to
# uniform scores when the markov chain is not ergodic.
@pytest.mark.parametrize("algorithm", [fit_rc])
def test_no_features_nonergodic_rankcentrality(algorithm):
    n_items = 3
    comps = np.array([[0, 1], [1, 2]])
    comp_results = np.array([0, 0])
    scores = algorithm(PairwiseData(n_items, comps, comp_results))
    assert np.allclose(scores, 1 / 3)


@flaky
@pytest.mark.parametrize(
    "algorithm",
    [fit_rc_diff, fit_rc_diff_decayed, fit_ranksvm, fit_ranksvm_rf, fit_siamese],
)
def test_features(algorithm):
    n_items = 3
    comps = np.array([[0, 1], [0, 2], [2, 1]])
    comp_results = np.array([0, 0, 1])
    features = np.array([[0, 0], [0.1, 0.1], [0.2, 0.2]])
    scores = algorithm(PairwiseData(n_items, comps, comp_results), features)
    assert scores[0] > scores[1]
    assert scores[1] > scores[2]
