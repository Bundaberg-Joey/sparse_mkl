import pytest
import numpy as np

from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.datasets import make_regression

from mkl.acquisition import GreedyNRanking
from mkl.data import Hdf5Dataset
from mkl.kernel import RbfKernelIdx, WhiteKernelIdx, TanimotoKernelIdx


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
    indices = np.random.choice(len(posterior), size=n_opt, replace=False)
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
# test that the kernel output matches one done by the "regular kernel"

@pytest.mark.parametrize('n, m', [
    (1, 2),
    (3, 4),
    (10, 10),
    (10, 1),
    (1, 10)
])
def test_RbfKernelIdx(n, m):
    ref, _ = make_regression(n_samples=100, n_features=5)
    a = np.random.randint(0, 99, size=n)
    b = np.random.randint(0, 99, size=m)
    
    kernel = RBF(length_scale=np.ones(ref.shape[1]))
    idx_kernel = RbfKernelIdx(dataset=ref, length_scale=np.ones(ref.shape[1]))

    kernel_outs = [
        kernel(ref[a]),
        kernel(ref[b]),
        kernel(ref[a], ref[a]),
        kernel(ref[a], ref[b]),
        kernel(ref[b], ref[a]),
        kernel(ref[b], ref[b])
    ]

    idx_kernel_outs = [
        idx_kernel(a),
        idx_kernel(b),
        idx_kernel(a, a),
        idx_kernel(a, b),
        idx_kernel(b, a),
        idx_kernel(b, b)
    ]
    
    for xa, xb in zip(kernel_outs, idx_kernel_outs):
        assert np.array_equal(xa, xb)
        
    with pytest.raises(IndexError):
        idx_kernel([len(ref)+1])
    


@pytest.mark.parametrize('n, m', [
    (1, 2),
    (3, 4),
    (10, 10),
    (10, 1),
    (1, 10)
])
def test_WhiteKernelIdx(n, m):
    ref, _ = make_regression(n_samples=100, n_features=5)
    a = np.random.randint(0, 99, size=n)
    b = np.random.randint(0, 99, size=m)
    
    kernel = WhiteKernel()
    idx_kernel = WhiteKernelIdx(dataset=ref)

    kernel_outs = [
        kernel(ref[a]),
        kernel(ref[b]),
        kernel(ref[a], ref[a]),
        kernel(ref[a], ref[b]),
        kernel(ref[b], ref[a]),
        kernel(ref[b], ref[b])
    ]

    idx_kernel_outs = [
        idx_kernel(a),
        idx_kernel(b),
        idx_kernel(a, a),
        idx_kernel(a, b),
        idx_kernel(b, a),
        idx_kernel(b, b)
    ]
    
    for xa, xb in zip(kernel_outs, idx_kernel_outs):
        assert np.array_equal(xa, xb)
        
    with pytest.raises(IndexError):
        idx_kernel([len(ref)+1])


def test_TanimotoKernelIdx__calc_sim():
    X = np.array([[1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1]]).astype(bool)  # convert to silence sklearn warning
    out = TanimotoKernelIdx._calc_sim(X, X, n_jobs=1)

    assert all(np.diag(out) == 1), 'diagonal should be 1'
    assert out.shape == (X.shape[0], X.shape[0]), 'should be square matrix w.r.t number of entries in X.'

    expected = np.array([[1., 0.33333333, 0.66666667],
                         [0.33333333, 1., 0.66666667],
                         [0.66666667, 0.66666667, 1.]])

    assert abs((out - expected)).sum() < 1e-7, 'Arrays should match, difference here due to floating point precision.'


@pytest.mark.parametrize("x, y, eval_gradient", 
[
    (np.array([0, 1, 4, 5]), None, False),
    (np.array([0, 1, 4, 5]).reshape(-1, 1), None, False),
    (np.array([0, 1, 4, 5]), None, True),
    (np.array([0, 1, 4, 5]).reshape(-1, 1), None, True),
    (np.array([0, 1, 4, 5]), np.arange(10), False),
    (np.arange(10), np.array([0, 1, 4, 5]), False),
    (np.arange(10), np.array([0, 1, 4, 5]).reshape(-1, 1), False),
    (np.arange(10).reshape(-1, 1), np.array([0, 1, 4, 5]).reshape(-1, 1), True),
    (np.arange(10), None, False),
    (np.arange(10).reshape(-1, 1), None, True)
])
def test_DynamicTanimotoKernel___call__(x, y, eval_gradient):
    fm = np.random.randint(0, 2, size=(30, 5), dtype=int)
    kernel = TanimotoKernelIdx(fm, 1)
    out = kernel(x, y, eval_gradient)

    if eval_gradient:
        covar, grad = out
        assert grad.shape == (len(x), len(x), 0)
        assert grad.sum() == 0
    else:
        covar = np.array(out)

    xr = x.ravel()
    yr = y.ravel() if y is not None else xr

    assert covar.ndim == 2
    assert covar.shape == (len(xr), len(yr))

    ref = kernel._calc_sim(fm[xr], fm[yr], 1)

    np.testing.assert_allclose(covar, ref)

# -----------------------------------------------------------------------------------------------------------------------------
