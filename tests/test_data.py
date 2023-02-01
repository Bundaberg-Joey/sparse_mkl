import pytest
import numpy as np

from mkl.data import Hdf5Dataset
from mkl.dense import DenseRBFModel, DenseTanimotoModel, DynamicDenseMKL


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

@pytest.mark.slow
@pytest.mark.parametrize("n, model, dataset", 
[
                             (1, DenseRBFModel, 'COF_p.hdf5'),
                             (10, DenseRBFModel ,'COF_p.hdf5'),
                             (1, DenseTanimotoModel, 'molecule.hdf5'),
                             (10, DenseTanimotoModel ,'molecule.hdf5'),
])
def test_hdf5_with_dense_models(n, model, dataset):
    X = Hdf5Dataset(F'tests/data/{dataset}', 'X')
    y = Hdf5Dataset(F'tests/data/{dataset}', 'y')[:].ravel()  # easier than loading from pandas
    
    inducing_indices = RAND.choice(len(X), size=50, replace=False)
    model = model(X=X, X_M=inducing_indices)
    
    train_indices = RAND.choice([i for i in range(len(X)) if i not in inducing_indices], size=n)
    y_train = y[train_indices]
    
    model.fit(train_indices, y_train)
    _ = model.calc_k_xm()
    _  = model.calc_k_mm()
    # just check they work, dont really care about the output
    

@pytest.mark.slow
@pytest.mark.parametrize("n", [1, 10])
def test_hdf5_with_dynamic_dense_model(n):
    X = Hdf5Dataset(F'tests/data/molecule.hdf5', 'X')
    y = Hdf5Dataset(F'tests/data/molecule.hdf5', 'y')[:].ravel()  # easier than loading from pandas
    
    inducing_indices = RAND.choice(len(X), size=50, replace=False)
    model = DynamicDenseMKL(X=[X, X], X_M=[inducing_indices, inducing_indices])
    
    train_indices = RAND.choice([i for i in range(len(X)) if i not in inducing_indices], size=n)
    y_train = y[train_indices]
    
    model.fit(train_indices, y_train)
    _ = model.calc_k_xm()
    _  = model.calc_k_mm()
    # just check they work, dont really care about the output


# -----------------------------------------------------------------------------------------------------------------------------
