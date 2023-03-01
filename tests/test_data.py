import pytest
import numpy as np

from mkl.data import Hdf5Dataset
from mkl.dense import DenseGaussianProcessregressor

# -----------------------------------------------------------------------------------------------------------------------------

RAND = np.random.RandomState(1)

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
    assert dataset.shape == (20, 5)
    assert len(dataset) == 20
    
    out = dataset[indices]
    
    assert out.ndim == 2
    assert out.shape == (len(indices), m)
    assert np.array_equal(out, ref[np.ravel(indices)])
    
    
# -----------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 10]) 
def test_hdf5_with_dense_models(n):
    X = Hdf5Dataset(F'tests/data/COF_p.hdf5', 'X')
    y = Hdf5Dataset(F'tests/data/COF_p.hdf5', 'y')[:].ravel()  # easier than loading from pandas
    
    model = DenseGaussianProcessregressor(data_set=X)
    
    train_indices = RAND.choice(len(X), size=n)
    y_train = y[train_indices]
    
    model.fit(train_indices, y_train)
    
    for i in range(1, 4):
        post = model.sample_y(n_samples=i)
        assert post.shape == (len(X), i)
    
    # just check they work, dont really care about the output
    

# -----------------------------------------------------------------------------------------------------------------------------
