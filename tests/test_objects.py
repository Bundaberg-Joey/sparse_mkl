import pytest
import numpy as np

from mkl.acquisition import GreedyNRanking
from mkl.data import Hdf5Dataset


# -----------------------------------------------------------------------------------------------------------------------------

RAND = np.random.RandomState(1)

# -----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize('n_samp, n_opt', 
                         [
                             (5, 20),  # random numbers
                             (100, 5),
                             (20, 1),
                         ])
def test_GreedyNRanking(n_samp, n_opt):
    n_samp = 5
    n_opt = 20
    posterior = np.ones((1000, n_samp))
    indices = RAND.choice(len(posterior), size=n_opt, replace=False)
    posterior[indices] *= 10.0  # boost these posterior values
    
    greedy = GreedyNRanking(n_opt)
    assert greedy.n_opt == n_opt
    
    scores = greedy.score_points(posterior)
    assert scores.shape == (len(posterior),)
    
    marked_alpha = np.sort(np.argsort(scores)[-n_opt:])
    match = np.array_equal(np.sort(marked_alpha), np.sort(indices))

    assert match, 'Marked indices match determined indices.'
    assert len(marked_alpha) == n_opt, 'All indices identified (prevents matching on empty arrays).'
    
    
# -----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("indices", 
                         [
                             [0, 5, 2, 10],
                            [[0], [5], [2], [10]],
                             [19, 18, 17],
                             list(range(10)),
                             [1]
                         ])
def test_Hdf5Dataset(indices):
    m = 5
    ref = np.arange(100).reshape(20, m)
    
    dataset = Hdf5Dataset('tests/test.hdf5')
    out = dataset[indices]
    
    assert out.ndim == 2
    assert out.shape == (len(indices), m)
    assert np.array_equal(out, ref[np.ravel(indices)])
    
    
# -----------------------------------------------------------------------------------------------------------------------------
