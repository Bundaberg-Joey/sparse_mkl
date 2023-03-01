import pytest

import pandas as pd
from sklearn.preprocessing import scale
import numpy as np

from mkl.dense import DenseGaussianProcessregressor
from mkl.acquisition import GreedyNRanking
from mkl.data import Hdf5Dataset


# -----------------------------------------------------------------------------------------------------------------------------


RAND = np.random.RandomState(1)


# -----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_sparse_gp_screening_rbf():
    df = pd.read_csv('tests/data/COF_pct_deliverablecapacityvSTPv.csv')
    subsample = RAND.choice(len(df), size=1000, replace=False)
    X, y = df.iloc[subsample, :5].values, df.iloc[subsample, -1].values
    X = scale(X)

    n_top = 100
    y_top = np.argsort(y)[-n_top:]

    # ----------------------------------------------------
    X_train_ind = list(RAND.choice(len(X), size=50))
    y_train = y[X_train_ind]

    # ----------------------------------------------------
    model = DenseGaussianProcessregressor(data_set=X)
    acqu = GreedyNRanking()

    for itr in range(50, 100):
            
        y_train = y[X_train_ind]
        
        model.fit(X_train_ind, y_train)
        posterior = model.sample_y(n_samples=1)
        
        alpha = acqu.score_points(posterior)
        alpha_ranked = np.argsort(alpha)[::-1]
        to_sample = [i for i in alpha_ranked if i not in X_train_ind][0]
        
        X_train_ind.append(to_sample)    
        n_top_found = sum([1 for i in X_train_ind if i in y_top])
        
        print(F'[{itr} / 100] : {n_top_found}/{n_top} found : y_max_tested={max(y):.3f} : y_sampled={y[to_sample]}')
        
        if n_top_found == 25:
            break
        
    assert n_top_found >= 25
    

# -----------------------------------------------------------------------------------------------------------------------------
