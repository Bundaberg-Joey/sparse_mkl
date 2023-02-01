import pytest
import numpy as np

from mkl.data import Hdf5Dataset


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
    
    dataset = Hdf5Dataset('tests/data/test.hdf5')
    out = dataset[indices]
    
    assert out.ndim == 2
    assert out.shape == (len(indices), m)
    assert np.array_equal(out, ref[np.ravel(indices)])
    
    
# -----------------------------------------------------------------------------------------------------------------------------
