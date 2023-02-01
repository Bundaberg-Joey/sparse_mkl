import pytest
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq

from mkl.data import Hdf5Dataset
from mkl.acquisition import GreedyNRanking
from mkl.sparse import SparseGaussianProcess, EnsembleSparseGaussianProcess
from mkl.dense import DenseRBFModel


# -----------------------------------------------------------------------------------------------------------------------------

RAND = np.random.RandomState(1)

# -----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_ensemble_rbf_hdf5():
    # just use two rbf models for the ensemble 
    X = Hdf5Dataset('tests/data/COF_p.hdf5', 'X')
    y = Hdf5Dataset('tests/data/COF_p.hdf5', 'y')[:].ravel()

    n_top = 100
    y_top = np.argsort(y)[-n_top:]

    # ----------------------------------------------------
    X_train_ind = list(RAND.choice(len(X), size=50))
    y_train = y[X_train_ind]


    # ----------------------------------------------------
    kms = KMeans(n_clusters=300, max_iter=5)
    kms.fit(X[:])
    X_cls = kms.cluster_centers_
    cls_ind, _ = vq(X_cls, X[:])  # need to use in data locations

    # ----------------------------------------------------
    dense_rbf_model = DenseRBFModel(X=X, X_M=cls_ind)
    sparse_rbf_model = SparseGaussianProcess(dense_model=dense_rbf_model)
    model = EnsembleSparseGaussianProcess(sparse_rbf_model, sparse_rbf_model, n_top=10, n_max=40)
    acqu = GreedyNRanking()

    for itr in range(50, 100):
            
        y_train = y[X_train_ind]

        model.fit(X_train_ind, y_train)

        posterior = model.sample_y(n_samples=1)
        _, _  = model.predict()
        alpha = acqu.score_points(posterior)
        alpha_ranked = np.argsort(alpha)[::-1]
        to_sample = [i for i in alpha_ranked if i not in X_train_ind][0]
        
        X_train_ind.append(to_sample)    
        n_top_found = sum([1 for i in X_train_ind if i in y_top])
        
        print(F'[{itr} / 100] : {n_top_found}/{n_top} found : y_max_tested={max(y):.3f} : y_sampled={y[to_sample]}')


# -----------------------------------------------------------------------------------------------------------------------------

