import pytest

import pandas as pd
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import numpy as np
from scipy.cluster.vq import vq

from mkl.sparse import SparseGaussianProcess
from mkl.dense import DenseRBFModel, DenseTanimotoModel, DenseMultipleKernelLearner, DynamicDenseMKL
from mkl.acquisition import GreedyNRanking


# -----------------------------------------------------------------------------------------------------------------------------


RAND = np.random.RandomState(1)


# -----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_sparse_gp_screening_rbf():
    df = pd.read_csv('tests/COF_pct_deliverablecapacityvSTPv.csv')
    subsample = RAND.choice(len(df), size=1000, replace=False)
    X, y = df.iloc[subsample, :5].values, df.iloc[subsample, -1].values
    X = scale(X)

    n_top = 100
    y_top = np.argsort(y)[-n_top:]

    # ----------------------------------------------------
    X_train_ind = list(RAND.choice(len(X), size=50))
    y_train = y[X_train_ind]


    # ----------------------------------------------------
    kms = KMeans(n_clusters=300, max_iter=5)
    kms.fit(X)
    X_cls = kms.cluster_centers_
    cls_ind, _ = vq(X_cls, X)  # need to use in data locations

    # ----------------------------------------------------


    dense_model = DenseRBFModel(X=X, X_M=cls_ind)
    model = SparseGaussianProcess(dense_model=dense_model)
    acqu = GreedyNRanking()

    for itr in range(50, 100):
            
        y_train = y[X_train_ind]

        if itr % 10 == 0:
            model.update_parameters(X_train_ind, y_train)
        model.update_data(X_train_ind, y_train)

        posterior = model.sample_y(n_samples=1)
        _, _  = model.predict()
        alpha = acqu.score_points(posterior)
        alpha_ranked = np.argsort(alpha)[::-1]
        to_sample = [i for i in alpha_ranked if i not in X_train_ind][0]
        
        X_train_ind.append(to_sample)    
        n_top_found = sum([1 for i in X_train_ind if i in y_top])
        
        print(F'[{itr} / 100] : {n_top_found}/{n_top} found : y_max_tested={max(y):.3f} : y_sampled={y[to_sample]}')
        
        
    assert n_top_found >= 25
    

# -----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_sparse_gp_screening_tanimoto():
    df = pd.read_csv('tests/molecule.csv')
    X, y = df.iloc[:, :2048].values, df.iloc[:, -1].values

    n_top = 50
    y_top = np.argsort(y)[-n_top:]

    # ----------------------------------------------------
    X_train_ind = list(RAND.choice(len(X), size=10))
    y_train = y[X_train_ind]

    # ----------------------------------------------------
    kms = KMeans(n_clusters=20, max_iter=5)
    kms.fit(X)
    X_cls = kms.cluster_centers_
    cls_ind, _ = vq(X_cls, X)  # need to use in data locations

    # ----------------------------------------------------
    dense_model = DenseTanimotoModel(X=X, X_M=cls_ind)
    model = SparseGaussianProcess(dense_model=dense_model)
    acqu = GreedyNRanking()

    for itr in range(10, 51):
            
        y_train = y[X_train_ind]

        if itr % 10 == 0:
            model.update_parameters(X_train_ind, y_train)
        model.update_data(X_train_ind, y_train)

        posterior = model.sample_y(n_samples=1)
        _, _  = model.predict()
        alpha = acqu.score_points(posterior)
        alpha_ranked = np.argsort(alpha)[::-1]
        to_sample = [i for i in alpha_ranked if i not in X_train_ind][0]
        
        X_train_ind.append(to_sample)    
        n_top_found = sum([1 for i in X_train_ind if i in y_top])
        
        print(F'[{itr} / 50] : {n_top_found}/{n_top} found : y_max_tested={max(y):.3f} : y_sampled={y[to_sample]}')
        
        
    assert n_top_found >= 17  # average with random guessing is 6.5
   
   
# -----------------------------------------------------------------------------------------------------------------------------

    
@pytest.mark.slow
@pytest.mark.parametrize("model", 
                         [
                             DenseMultipleKernelLearner,
                             DynamicDenseMKL
                         ])
def test_sparse_gp_screening_dense_mkl_models_double_same_kernel(model):
    # just double up the same kernel used for the singular tanimoto approach (test is the same but need to pass different datasets at init)
    df = pd.read_csv('tests/molecule.csv')
    X, y = df.iloc[:, :2048].values, df.iloc[:, -1].values

    n_top = 50
    y_top = np.argsort(y)[-n_top:]

    # ----------------------------------------------------
    X_train_ind = list(RAND.choice(len(X), size=10))
    y_train = y[X_train_ind]

    # ----------------------------------------------------
    kms = KMeans(n_clusters=20, max_iter=5)
    kms.fit(X)
    X_cls = kms.cluster_centers_
    cls_ind, _ = vq(X_cls, X)  # need to use in data locations

    # ----------------------------------------------------
    dense_model = model(X=[X, X], X_M=[cls_ind, cls_ind])
    model = SparseGaussianProcess(dense_model=dense_model)
    acqu = GreedyNRanking()

    for itr in range(10, 51):
            
        y_train = y[X_train_ind]

        if itr % 10 == 0:
            model.update_parameters(X_train_ind, y_train)
        model.update_data(X_train_ind, y_train)

        posterior = model.sample_y(n_samples=1)
        _, _  = model.predict()
        alpha = acqu.score_points(posterior)
        alpha_ranked = np.argsort(alpha)[::-1]
        to_sample = [i for i in alpha_ranked if i not in X_train_ind][0]
        
        X_train_ind.append(to_sample)    
        n_top_found = sum([1 for i in X_train_ind if i in y_top])
        
        print(F'[{itr} / 50] : {n_top_found}/{n_top} found : y_max_tested={max(y):.3f} : y_sampled={y[to_sample]}')
        
        
    assert n_top_found >= 17  # average with random guessing is 6.5
    

# -----------------------------------------------------------------------------------------------------------------------------
