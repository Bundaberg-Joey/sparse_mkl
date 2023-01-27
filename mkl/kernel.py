from typing import Union, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.gaussian_process.kernels import Kernel, RBF, WhiteKernel
from sklearn.utils import gen_even_slices

from joblib import Parallel, delayed, effective_n_jobs

from mkl.data import Hdf5Dataset


# -----------------------------------------------------------------------------------------------------------------------------


class TanimotoKernelIdx(Kernel):
    """Tanimoto kernel for integer arrays but accepts indices rather than explicit features.
    Explicit features are indexed from the database provided at init.
    Allows for easier compatability with other MKL models.
    """
    
    def __init__(self, dataset:Union[Hdf5Dataset, NDArray], n_jobs: int=1) -> None:
        self.dataset = dataset
        self.n_jobs = int(n_jobs)
    
    def __call__(self, X: NDArray[np.int_], Y: Optional[NDArray[np._int]]=None, eval_gradient: bool=False):
        Xa = self.dataset[X]
        
        if Y is None:
            K = self._calc_sim(Xa, Xa, self.n_jobs)
        else:
            Xb = self.dataset[Y]
            K = self._calc_sim(Xa, Xb, self.n_jobs)
            
        if eval_gradient:
            return K, np.zeros(shape=(len(X), len(X), 0))  # fixed params
        else:
            return K

    @staticmethod
    def _calc_sim(X: NDArray[NDArray[np.int_]], Y: NDArray[NDArray[np.int_]], n_jobs: int) -> NDArray[NDArray[np.float_]]:
        
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

    def diag(self, X: NDArray):
        return np.ones(len(X))  # tanimoto diagonal always 1

    def is_stationary(self):
        return True


# -----------------------------------------------------------------------------------------------------------------------------


class RbfKernelIdx(RBF):
    """Identical to `sklearn.gaussian_process.kernels.RBF` but accepts indices rather than explicit features.
    Explicit features are indexed from the database provided at init.
    Allows for easier compatability with other MKL models.
    """

    def __init__(self, dataset:Union[Hdf5Dataset, NDArray], length_scale: NDArray[np.float_]):
        self.dataset = dataset
        super().__init__(length_scale=length_scale)

    def __call__(self, X: NDArray[np.int_], Y: Optional[NDArray[np._int]]=None, eval_gradient: bool=False):
        Xa = self.dataset[X]

        if Y is None:
            return super().__call__(Xa, Y, eval_gradient)
        else:
            Xb = self.dataset[Y]
            return super().__call__(Xa, Xb, eval_gradient)

    def diag(self, X: NDArray):
        Xa = self.dataset[X]
        return super().diag(Xa)


# -----------------------------------------------------------------------------------------------------------------------------


class WhiteKernelIdx(WhiteKernel):
    """Identical to `sklearn.gaussian_process.kernels.WhiteKernel` but accepts indices rather than explicit features.
    Explicit features are indexed from the database provided at init.
    Allows for easier compatability with other MKL models.
    """
    
    def __init__(self, dataset:Union[Hdf5Dataset, NDArray], noise_level: float=1.0):
        self.dataset = dataset
        super().__init__(noise_level=noise_level)
        
    def __call__(self, X, Y=None, eval_gradient=False):
        X = self.dataset[X]
        
        if Y is None:
            return super().__call__(X, Y, eval_gradient)
        else:
            Y = X if Y is None else self.dataset[Y]
            return super().__call__(X, Y, eval_gradient)
    
    def diag(self, X):
        X = self.dataset[X]
        return super().diag(X)
    
    
    # -----------------------------------------------------------------------------------------------------------------------------
