
import numpy as np
from sklearn.gaussian_process.kernels import Kernel, RBF
from sklearn.utils import gen_even_slices

from joblib import Parallel, delayed, effective_n_jobs



def calc_tanimoto_similarity(X, Y, n_jobs):
    
        def _dist_wrapper(dist_func, dist_matrix, slice_, *args, **kwargs):
            #Taken from `sklearn.metrics.pairwise` without modification.
            dist_matrix[:, slice_] = dist_func(*args, **kwargs)

        def _tanimoto(Xa, Xb):
            # Xa, Xb must be integer types!
            c = np.einsum('ij,kj->ik', Xa, Xb)
            a = Xa.sum(1).reshape(-1, 1)
            b = Xb.sum(1)
            return c / (a + b - c)

        X = X.astype(int) if X.dtype != np.int_ else X
        Y = Y.astype(int) if Y.dtype != np.int_ else Y

        fd = delayed(_dist_wrapper)
        n_jobs = effective_n_jobs(n_jobs)
        p = Parallel(backend="threading", n_jobs=n_jobs)
        K = np.empty((X.shape[0], Y.shape[0]), dtype=float, order="F")
        p(fd(_tanimoto, K, s, X, Y[s]) for s in gen_even_slices(len(Y), n_jobs))
        return K



class TanimotoKernel(Kernel):
    
    def __init__(self, n_jobs) -> None:
    
    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            K = self._calc_tanimoto_similarity(X, X, self.n_jobs)
        else:
            K = self._calc_tanimoto_similarity(X, Y, self.n_jobs)
        
        if eval_gradient:
            n = len(X)
            return K, np.zeros(shape=(n, n, 0))  # fixed hyper parameters
        else:
            return K

    def diag(self, X):
        return np.ones(len(X))  # tanimoto diagonal always 1

    def is_stationary(self):
        return True


class RbfKernel(RBF):
    """For consistent API during screening, this class is the same as `sklearn.gaussian_process.kernels.RBF` but is
    set up to accept index arrays rather than the explicit features themselves.
    This is identical to how the other Tanimoto kernels are run.
    """

    def __init__(self, X_rbf, length_scale):
        # store the feature matrix
        self.X_rbf = X_rbf
        super().__init__(length_scale=length_scale)

    def __call__(self, X, Y=None, eval_gradient=False):
        # slices stored array and passes it to kernel call via super.
        Xa = slice_features(self.X_rbf, X)

        if Y is None:
            return super().__call__(Xa, Y, eval_gradient)
        else:
            Xb = slice_features(self.X_rbf, Y)
            return super().__call__(Xa, Xb, eval_gradient)

    def diag(self, X):
        # slicing behaviour as with `__call__`
        Xa = slice_features(self.X_rbf, X)
        return super().diag(Xa)
