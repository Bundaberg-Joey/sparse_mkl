from time import perf_counter

import pytest
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.datasets import make_regression

from mkl.acquisition import GreedyNRanking
from mkl.data import Hdf5Dataset
from mkl.kernel import RbfKernelIdx, WhiteKernelIdx, TanimotoKernelIdx
from mkl._model_old import SparseGaussianProcess


# -----------------------------------------------------------------------------------------------------------------------------

RAND = np.random.RandomState(1)

# -----------------------------------------------------------------------------------------------------------------------------


# @pytest.mark.parametrize('n_samp, n_opt', 
#                          [
#                              (5, 20),  # random numbers
#                              (100, 5),
#                              (20, 1),
#                          ])
# def test_GreedyNRanking(n_samp, n_opt):
#     n_samp = 5
#     n_opt = 20
#     posterior = np.ones((1000, n_samp))
#     indices = RAND.choice(len(posterior), size=n_opt, replace=False)
#     posterior[indices] *= 10.0  # boost these posterior values
    
#     greedy = GreedyNRanking(n_opt)
#     assert greedy.n_opt == n_opt
    
#     scores = greedy.score_points(posterior)
#     assert scores.shape == (len(posterior),)
    
#     marked_alpha = np.sort(np.argsort(scores)[-n_opt:])
#     match = np.array_equal(np.sort(marked_alpha), np.sort(indices))

#     assert match, 'Marked indices match determined indices.'
#     assert len(marked_alpha) == n_opt, 'All indices identified (prevents matching on empty arrays).'
    
    
# # -----------------------------------------------------------------------------------------------------------------------------


# @pytest.mark.parametrize("indices", 
#                          [
#                              [0, 5, 2, 10],
#                             [[0], [5], [2], [10]],
#                              [19, 18, 17],
#                              list(range(10)),
#                              [1]
#                          ])
# def test_Hdf5Dataset(indices):
#     m = 5
#     ref = np.arange(100).reshape(20, m)
    
#     dataset = Hdf5Dataset('tests/test.hdf5')
#     out = dataset[indices]
    
#     assert out.ndim == 2
#     assert out.shape == (len(indices), m)
#     assert np.array_equal(out, ref[np.ravel(indices)])
    
    
# # -----------------------------------------------------------------------------------------------------------------------------
# # test that the kernel output matches one done by the "regular kernel"

# @pytest.mark.parametrize('n, m', [
#     (1, 2),
#     (3, 4),
#     (10, 10),
#     (10, 1),
#     (1, 10)
# ])
# def test_RbfKernelIdx(n, m):
#     ref, _ = make_regression(n_samples=100, n_features=5)
#     a = RAND.randint(0, 99, size=n)
#     b = RAND.randint(0, 99, size=m)
    
#     kernel = RBF(length_scale=np.ones(ref.shape[1]))
#     idx_kernel = RbfKernelIdx(dataset=ref, length_scale=np.ones(ref.shape[1]))

#     kernel_outs = [
#         kernel(ref[a]),
#         kernel(ref[b]),
#         kernel(ref[a], ref[a]),
#         kernel(ref[a], ref[b]),
#         kernel(ref[b], ref[a]),
#         kernel(ref[b], ref[b])
#     ]

#     idx_kernel_outs = [
#         idx_kernel(a),
#         idx_kernel(b),
#         idx_kernel(a, a),
#         idx_kernel(a, b),
#         idx_kernel(b, a),
#         idx_kernel(b, b)
#     ]
    
#     for xa, xb in zip(kernel_outs, idx_kernel_outs):
#         assert np.array_equal(xa, xb)
        
#     with pytest.raises(IndexError):
#         idx_kernel([len(ref)+1])
    


# @pytest.mark.parametrize('n, m', [
#     (1, 2),
#     (3, 4),
#     (10, 10),
#     (10, 1),
#     (1, 10)
# ])
# def test_WhiteKernelIdx(n, m):
#     ref, _ = make_regression(n_samples=100, n_features=5)
#     a = RAND.randint(0, 99, size=n)
#     b = RAND.randint(0, 99, size=m)
    
#     kernel = WhiteKernel()
#     idx_kernel = WhiteKernelIdx(dataset=ref)

#     kernel_outs = [
#         kernel(ref[a]),
#         kernel(ref[b]),
#         kernel(ref[a], ref[a]),
#         kernel(ref[a], ref[b]),
#         kernel(ref[b], ref[a]),
#         kernel(ref[b], ref[b])
#     ]

#     idx_kernel_outs = [
#         idx_kernel(a),
#         idx_kernel(b),
#         idx_kernel(a, a),
#         idx_kernel(a, b),
#         idx_kernel(b, a),
#         idx_kernel(b, b)
#     ]
    
#     for xa, xb in zip(kernel_outs, idx_kernel_outs):
#         assert np.array_equal(xa, xb)
        
#     with pytest.raises(IndexError):
#         idx_kernel([len(ref)+1])


# def test_TanimotoKernelIdx__calc_sim():
#     X = np.array([[1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1]]).astype(bool)  # convert to silence sklearn warning
#     out = TanimotoKernelIdx._calc_sim(X, X, n_jobs=1)

#     assert all(np.diag(out) == 1), 'diagonal should be 1'
#     assert out.shape == (X.shape[0], X.shape[0]), 'should be square matrix w.r.t number of entries in X.'

#     expected = np.array([[1., 0.33333333, 0.66666667],
#                          [0.33333333, 1., 0.66666667],
#                          [0.66666667, 0.66666667, 1.]])

#     assert abs((out - expected)).sum() < 1e-7, 'Arrays should match, difference here due to floating point precision.'


# @pytest.mark.parametrize("x, y, eval_gradient", 
# [
#     (np.array([0, 1, 4, 5]), None, False),
#     (np.array([0, 1, 4, 5]).reshape(-1, 1), None, False),
#     (np.array([0, 1, 4, 5]), None, True),
#     (np.array([0, 1, 4, 5]).reshape(-1, 1), None, True),
#     (np.array([0, 1, 4, 5]), np.arange(10), False),
#     (np.arange(10), np.array([0, 1, 4, 5]), False),
#     (np.arange(10), np.array([0, 1, 4, 5]).reshape(-1, 1), False),
#     (np.arange(10).reshape(-1, 1), np.array([0, 1, 4, 5]).reshape(-1, 1), True),
#     (np.arange(10), None, False),
#     (np.arange(10).reshape(-1, 1), None, True)
# ])
# def test_DynamicTanimotoKernel___call__(x, y, eval_gradient):
#     fm = RAND.randint(0, 2, size=(30, 5), dtype=int)
#     kernel = TanimotoKernelIdx(fm, 1)
#     out = kernel(x, y, eval_gradient)

#     if eval_gradient:
#         covar, grad = out
#         assert grad.shape == (len(x), len(x), 0)
#         assert grad.sum() == 0
#     else:
#         covar = np.array(out)

#     xr = x.ravel()
#     yr = y.ravel() if y is not None else xr

#     assert covar.ndim == 2
#     assert covar.shape == (len(xr), len(yr))

#     ref = kernel._calc_sim(fm[xr], fm[yr], 1)

#     np.testing.assert_allclose(covar, ref)

# # -----------------------------------------------------------------------------------------------------------------------------


# def test_SparseGaussianProcess_output_sizes_regular():
#     #confirm works wither "regular" non indexed kernels
#     X, y = make_regression(n_samples=100, n_features=5, random_state=1)
    
#     kernel = RBF(length_scale=np.ones(X.shape[1])) + WhiteKernel()
#     internal_model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
#     model = SparseGaussianProcess(model=internal_model, X_inducing=X[:10])
    
#     model.fit(X[:20], y[:20])
    
#     for n in (50, 80):
#         mu, std = model.predict(X[:n], return_std=True)
        
#         assert mu.shape == (n, )
#         assert std.shape == (n, )
        
#         for m in (1, 2):
#             posterior = model.sample_y(X[:n], n_samples=m)
#             assert posterior.shape == (n, m)
        

# def test_SparseGaussianProcess_output_sizes_indexed():
#     #confirm works with indeexed kernels
#     X, y = make_regression(n_samples=100, n_features=5, random_state=1)
#     X_range = np.arange(len(X)).reshape(-1, 1)
    
#     kernel = RbfKernelIdx(X, length_scale=np.ones(X.shape[1])) + WhiteKernelIdx(X)
#     internal_model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
#     model = SparseGaussianProcess(model=internal_model, X_inducing=X_range[:10])
    
#     model.fit(X_range[:20], y[:20])
    
#     for n in (50, 80):
#         mu, std = model.predict(X_range[:n], return_std=True)
        
#         assert mu.shape == (n, )
#         assert std.shape == (n, )
#         assert not np.isnan(mu).any()
#         assert not np.isnan(std).any()
        
#         for m in (1, 2):
#             posterior = model.sample_y(X_range[:n], n_samples=m)
#             assert posterior.shape == (n, m)
#             assert not np.isnan(posterior).any()
#             assert not np.isnan(posterior).any()
            
            
# def test_SparseGaussianProcess_TanimotoIdx():
#     #confirm works with indeexed kernels
    
#     X = RAND.randint(0, 2, size=(100, 5), dtype=int)
#     y = RAND.normal(140, 10, len(X))    
#     X_range = np.arange(len(X)).reshape(-1, 1)
    
#     kernel = TanimotoKernelIdx(X) + WhiteKernelIdx(X)
#     internal_model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
#     model = SparseGaussianProcess(model=internal_model, X_inducing=X_range[:10])
    
#     model.fit(X_range[:20], y[:20])
    
#     for n in (50, 80):
#         mu, std = model.predict(X_range[:n], return_std=True)
        
#         assert mu.shape == (n, )
#         assert std.shape == (n, )
#         assert not np.isnan(mu).any()
#         assert not np.isnan(std).any()
        
#         for m in (1, 2):
#             posterior = model.sample_y(X_range[:n], n_samples=m)
#             assert posterior.shape == (n, m)
#             assert not np.isnan(posterior).any()
#             assert not np.isnan(posterior).any()


# @pytest.mark.slow
# def test_SparseGaussianProcess_runs_quickly_on_large():
#     X = RAND.randn(100_000, 5)
#     y = RAND.randn(100_000)
#     X_ind = X[-100:]

#     kernel = RBF(length_scale=np.ones(5)) + WhiteKernel()
#     internal_model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
#     model = SparseGaussianProcess(internal_model, X_inducing=X_ind)

#     a = perf_counter()
#     # about 5-8 seconds to fit to 1_000 and sample for 100_000 and inducing matrix of 100
#     model.fit(X[:1000], y[:1000])
#     model.sample_y(X, n_samples=100)
#     b = perf_counter()

#     time_taken = b - a
#     assert time_taken <= 20


def test_compare_SparseGaussianProcess_against_regular():
    X, y = make_regression(n_samples=100, n_features=5, noise=1.0, random_state=1)
    X_range = np.arange(len(X)).reshape(-1, 1)
    
    kernel = RbfKernelIdx(X, length_scale=np.ones(X.shape[1])) + WhiteKernelIdx(X)
    internal_model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

    model = SparseGaussianProcess(model=internal_model, X_inducing=X_range[:5])    
    ref_model = GaussianProcessRegressor(kernel=RBF(length_scale=np.ones(X.shape[1])) + WhiteKernel(), normalize_y=True)
    
    model.fit(X_range[:5], y[:5])
    ref_model.fit(X[:5], y[:5])
    
    print(model.model.kernel_)  # both values match between kernels so definitely being "fit"
    print(ref_model.kernel_)
    
    print(model._get_kernel_variance())
    
    mu_m, std_m = model.predict(X_range, return_std=True)  # works
    mu_r, std_r = ref_model.predict(X, return_std=True)

    
    #MODEL IS JUST RETURNING THE MEAN VALUE WITH SOME NOISE ATTACHED TO IT!
    print()
    print('y_train: ', y[:5])
    print()
    print('y_train mean: ', y[:5].mean())
    print()
    print('sparse_mu: ',mu_m[:10])    
    print()
    print('ref_mu: ',mu_r[:10])    
    print()
    print('sparse: std',std_m[:10])    
    print()
    print('ref_std: ',std_r[:10])    
    print()




    # check varainces for known points are low
    
    
    
    
def test_if_normalize_y_should_be_used_or_not():
    pass
