import pytest
import numpy as np

import rankcentrality as rc


@pytest.mark.parametrize("n_items,n_comps", [(3, 100), (8, 200)])
def test_generates_comparisons(n_items, n_comps):
    comps, comp_results = rc.generate.simulate_comparisons(np.ones(n_items), n_comps)
    assert comps.shape == (n_comps, 2)
    assert comp_results.shape == (n_comps,)
    assert np.all(np.isin(comp_results, [0, 1]))
    assert np.all(np.isin(comps, np.arange(n_items)))
