from typing import Any, Tuple

import numpy as np

from rankcentrality.types import Scores, Comparisons, ComparisonResults, Matrix


def simulate_comparisons(
    scores: Scores, n_comps: int
) -> Tuple[Comparisons, ComparisonResults]:
    num_items = scores.shape[0]
    comps = np.random.randint(0, num_items, (n_comps, 2))
    comp_scores = scores[comps]
    pair_score_sum = comp_scores.sum(axis=1)
    # For those exceptional pairs where both points have score 0, we should
    # have a toss up
    comp_scores[pair_score_sum == 0, :] = 1
    pair_score_sum[pair_score_sum == 0] = 2
    prob_b_wins = comp_scores[:, 1] / pair_score_sum
    comp_results = np.random.binomial(1, prob_b_wins)
    return (comps, comp_results)
