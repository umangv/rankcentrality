from typing import Any, Tuple, Dict, List, NamedTuple
from nptyping import NDArray

Scores = NDArray[(Any,), float]

Comparisons = NDArray[(Any, Any), int]
ComparisonResults = NDArray[(Any,), int]

Matrix = NDArray[(Any, Any), float]

"""
An ExperimentResults object contains results for an experiment evaluated on a
specific dataset for which a true score vector w is known. In a given
experiment, there are
 - a set of algorithms that were evaluated,
 - a list of number of pairwise comparisons (to study the impact of number of
   pairwise comparisons), and
 - a number of repetitions for each (algorithm, number of comparisons) pair.

An object of type ExperimentResults is indexed as follows:
experiment_results[algorithm][num_comps_index][repetition_index]
is a score vector of the corresponding run of the experiment.
"""
ExperimentResults = Dict[str, List[List[Scores]]]


class Metrics(NamedTuple):
    means: NDArray[(Any,), float]
    std_errs: NDArray[(Any,), float]


"""An ExperimentMetrics object contains a loss for results in an experiment.

An object of type ExperimentMetrics is indexed as follows:
experiment_metrics[algorithm][metric][num_comps_index]
is the metric desired (where `metric` is either "means" or "std_errs").
"""
ExperimentMetrics = Dict[str, Metrics]
