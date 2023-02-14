import pytest
import numpy as np
from sklearn.cluster import KMeans
import gpflow
import tensorflow as tf

from mkl.data import Hdf5Dataset
from mkl.acquisition import GreedyNRanking
from mkl.sparse import SparseGaussianProcess, EnsembleSparseGaussianProcess
from mkl.dense import DenseRBFModel, DenseMatern12Model, DynamicDenseMKL


# -----------------------------------------------------------------------------------------------------------------------------

RAND = np.random.RandomState(1)
RBF_TOP = 25
TAN_TOP = 17

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
    M = kms.cluster_centers_

    # ----------------------------------------------------
    dense_rbf_model = DenseRBFModel(X=X, M=M)
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

        if n_top_found == RBF_TOP:
            break
        
    assert n_top_found >= RBF_TOP
# -----------------------------------------------------------------------------------------------------------------------------



@pytest.mark.slow

def test_ensemble_rbf_tan_hdf5():
    # This fails in this instance because the dataset is not that varied w.r.t to the rbf data
    # running in isolation of the other models (i.e. tanimoto) or as part of the ensemble below for this datase causes issues only with RBF Tanimoto is fine
    # Looking at this it seems that the kernels are prone to sensitivity issues
    # I scaled the dataset for the rbf features to unit variance and mean of 0 so that likely isnt the problem
    # see this link where matern kernels have a problem: https://github.com/GPflow/GPflow/issues/490
    # Using the new matern kernel helps a lot with this but doesnt eliminate the problem 100% but maybe about 75% for this crappy dataset
    # Use RBF for the final screening but if there are numerical issues then run with MAtern12
    X_rbf = Hdf5Dataset('tests/data/ens_rbf.hdf5', 'X')
    X_fp = Hdf5Dataset('tests/data/ens_fp.hdf5', 'X')
    y = Hdf5Dataset('tests/data/ens_fp.hdf5', 'y')[:].ravel()

    n_top = 50
    y_top = np.argsort(y)[-n_top:]

    # ----------------------------------------------------
    X_train_ind = list(RAND.choice(len(X_rbf), size=10))
    y_train = y[X_train_ind]


    # ----------------------------------------------------
    M_rbf = X_rbf[:][:20]
    M_fp = X_fp[:][:20]
    
    
    # ----------------------------------------------------
    dense_mat_model = DenseMatern12Model(X=X_rbf, M=M_rbf.copy())
    dense_mkl_model = DynamicDenseMKL(X=[X_fp, X_fp, X_fp], M=[M_fp.copy(), M_fp.copy(), M_fp.copy()])
    sparse_rbf_model = SparseGaussianProcess(dense_model=dense_mat_model)
    sparse_mkl_model = SparseGaussianProcess(dense_model=dense_mkl_model)
    
    model = EnsembleSparseGaussianProcess(sparse_rbf_model, sparse_mkl_model, n_top=10, n_max=40)
    acqu = GreedyNRanking()

    for itr in range(10, 51):
            
        y_train = y[X_train_ind]

        model.fit(X_train_ind, y_train)

        posterior = model.sample_y(n_samples=1)
        #_, _  = model.predict()
        alpha = acqu.score_points(posterior)
        alpha_ranked = np.argsort(alpha)[::-1]
        to_sample = [i for i in alpha_ranked if i not in X_train_ind][0]
        
        X_train_ind.append(to_sample)    
        n_top_found = sum([1 for i in X_train_ind if i in y_top])
        
        print(F'[{itr} / 50] : {n_top_found}/{n_top} found : y_max_tested={max(y):.3f} : y_sampled={y[to_sample]}')

        if n_top_found == TAN_TOP:
            break
        
    assert n_top_found >= TAN_TOP



@pytest.mark.slow
@pytest.mark.xfail
def test_rbf_hdf5():
    # single rbf instance of the above, expect to fail for the reasons doccumented there w.r.t to the data
    
    X_rbf = Hdf5Dataset('tests/data/ens_rbf.hdf5', 'X')
    y = Hdf5Dataset('tests/data/ens_rbf.hdf5', 'y')[:].ravel()

    n_top = 100
    y_top = np.argsort(y)[-n_top:]

    # ----------------------------------------------------
    X_train_ind = list(RAND.choice(len(X_rbf), size=50))
    y_train = y[X_train_ind]


    # ----------------------------------------------------
    M_rbf = X_rbf[:][:20]
    
    # ----------------------------------------------------
    dense_rbf_model = DenseRBFModel(X=X_rbf, M=M_rbf.copy())
    model = SparseGaussianProcess(dense_model=dense_rbf_model)
    
    acqu = GreedyNRanking()

    for itr in range(50, 100):
            
        y_train = y[X_train_ind]

        if itr % 10 == 0:
            model.update_parameters(X_train_ind, y_train)
        model.update_data(X_train_ind, y_train)

        posterior = model.sample_y(n_samples=1)
        #_, _  = model.predict()
        alpha = acqu.score_points(posterior)
        alpha_ranked = np.argsort(alpha)[::-1]
        to_sample = [i for i in alpha_ranked if i not in X_train_ind][0]
        
        X_train_ind.append(to_sample)    
        n_top_found = sum([1 for i in X_train_ind if i in y_top])
        
        print(F'[{itr} / 100] : {n_top_found}/{n_top} found : y_max_tested={max(y):.3f} : y_sampled={y[to_sample]}')

        if n_top_found == TAN_TOP:
            break
        
    assert n_top_found >= TAN_TOP





@pytest.mark.slow
def test_matern_hdf5():
    # single rbf instance of the above, expect to fail for the reasons doccumented there w.r.t to the data
    
    X_rbf = Hdf5Dataset('tests/data/ens_rbf.hdf5', 'X')
    y = Hdf5Dataset('tests/data/ens_rbf.hdf5', 'y')[:].ravel()

    n_top = 100
    y_top = np.argsort(y)[-n_top:]

    # ----------------------------------------------------
    X_train_ind = list(RAND.choice(len(X_rbf), size=50))
    y_train = y[X_train_ind]


    # ----------------------------------------------------
    M_rbf = X_rbf[:][:20]
    
    # ----------------------------------------------------
    dense_mat_model = DenseMatern12Model(X=X_rbf, M=M_rbf.copy())
    model = SparseGaussianProcess(dense_model=dense_mat_model)
    
    acqu = GreedyNRanking()

    for itr in range(10, 51):
            
        y_train = y[X_train_ind]

        if itr % 10 == 0:
            model.update_parameters(X_train_ind, y_train)
        model.update_data(X_train_ind, y_train)

        posterior = model.sample_y(n_samples=1)
        #_, _  = model.predict()
        alpha = acqu.score_points(posterior)
        alpha_ranked = np.argsort(alpha)[::-1]
        to_sample = [i for i in alpha_ranked if i not in X_train_ind][0]
        
        X_train_ind.append(to_sample)    
        n_top_found = sum([1 for i in X_train_ind if i in y_top])
        
        print(F'[{itr} / 50] : {n_top_found}/{n_top} found : y_max_tested={max(y):.3f} : y_sampled={y[to_sample]}')

        if n_top_found == TAN_TOP:
            break
        
    assert n_top_found >= TAN_TOP
