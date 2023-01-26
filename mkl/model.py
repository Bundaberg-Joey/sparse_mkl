from typing import List, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Kernel

# ----------------------------------------------------------------------------------------------------------------------


class SparseGaussianProcess:
    """Gaussian process capabale of performing sparse manipulations to speed up predictions / posterior samplng on large datasets.
    """

    def __init__(self, model: GaussianProcessRegressor, X_inducing: NDArray[np.int_], jitter: float=1e-6) -> None:
        """
        Parameters
        ----------
        model : GaussianProcessRegressor
            Instantiated sklearn.gaussian_process.GaussianProcessRegressor
            Kernel must accept indices instead of raw features.
            Also must have a sum kernel of `<Kernel>() + WhiteKernel()`
            
        X_inducing : NDArray[np.int_]
            Indices of MOFs to use for inducing matrix.
            
        jitter : float
            slight noise value to prevent numerical issues.
        """
        self.model = model
        self.X_inducing = np.array(X_inducing)  # indices of MOFs in inducing array 
        self.jitter = float(jitter)

    def fit(self, X_train: NDArray[np.int_], y_train: NDArray[np.float_]) -> None:
        """Fit model to passed data.

        Parameters
        ----------
        X_train : NDArray[np.int_]
            Indices of features to extract for kernel.
            
        y_train : NDArray[np.float_]
            performance values (not indices) of the passed MOF indices.
        """
        self.y_train = y_train
        self.model.fit(X_train, y_train)

        prior_var = y_train.std() ** 2
        self.kernel_var = self._get_kernel_variance()

        k_xm = self.model.kernel_(X_train, self.X_inducing)
        k_mm = self.model.kernel_(self.X_inducing, self.X_inducing)
        self.sig_xm_train = self.kernel_var * k_xm
        self.sig_mm_train = self.kernel_var * k_mm + np.identity(n=len(self.X_inducing)) * self.kernel_var * self.jitter
        self.updated_var = self.kernel_var + prior_var - np.sum(np.multiply(np.linalg.solve(self.sig_mm_train, self.sig_xm_train.T), self.sig_xm_train.T), 0)
        
    def _get_kernel_variance(self) -> float:
        """Get the kernel variance from the WhiteKernel in the kernel sum.
        the iteration saves having to specify `k2` directly which assumes that order of kernel operations in the sum is the same.
        
        sklearn returns log transformed values so `np.exp` used to transform back to linear values.

        Returns
        -------
        float
            kernel variance as found by the WhiteKernel
        """
        params = self.model.kernel_.get_params()
        noise_terms = [np.exp(params[k]) for k in params if 'noise_level' in k and 'bound' not in k]
        assert len(noise_terms) == 1
        return noise_terms[0]

    def _perform_sparse_manipulations(self, X: NDArray[np.int_], prior_mu: float) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """relevant matrix transformations / operations to achieve sparse GP

        Parameters
        ----------
        X : NDArray[np.int_]
            Indices of features to extract for kernel.
            Indices to use
            
        prior_mu : float
            mean of he prior distribution trget values.

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray, NDArray]
            transformed matrices.
        """

        k_xm_query = self.model.kernel(X, self.X_inducing)
        k_mm_query = self.model.kernel(self.X_inducing, self.X_inducing)

        sig_xm_query = self.kernel_var * k_xm_query
        sig_mm_query = self.kernel_var * k_mm_query + np.identity(len(self.X_inducing)) * self.jitter * self.kernel_var

        K = np.matmul(self.sig_xm_train.T, np.divide(self.sig_xm_train, self.updated_var.reshape(-1, 1)))
        sig_mm_pos = sig_mm_query - K + np.matmul(K, np.linalg.solve(K + sig_mm_query, K))
        J = np.matmul(self.sig_xm_train.T, np.divide(self.y_train - prior_mu, self.updated_var))
        mu_m_pos = prior_mu + J - np.matmul(K, np.linalg.solve(K + self.sig_mm_train, J))
        return  sig_xm_query, sig_mm_query, sig_mm_pos, mu_m_pos

    def predict(self, X: NDArray[np.int], return_std: bool=False) -> Union[Tuple[NDArray, NDArray], NDArray]:
        """Make predictions for passed MOFs

        Parameters
        ----------
        X : NDArray[np.int_]
            Indices of features to extract for kernel.
            
        return_std : bool, optional
            return the std of the model predictions (not calculated if value is False)

        Returns
        -------
        Union[Tuple[NDArray, NDArray], NDArray]
            Either two 1D arrays or a single array depending if `return_std` is specified.
        """
        prior_mu = self.y_train.mean()
        sig_xm_query, sig_mm_query, sig_mm_pos, mu_m_pos = self._perform_sparse_manipulations(X, prior_mu)
       
        mu_X_pos = prior_mu + np.matmul(sig_xm_query, np.linalg.solve(sig_mm_query, mu_m_pos - prior_mu))

        if return_std:
            var_X_pos = np.sum(np.multiply(np.matmul(np.linalg.solve(sig_mm_query, np.linalg.solve(sig_mm_query, sig_mm_pos).T), sig_xm_query.T), sig_xm_query.T), 0)
            return mu_X_pos, np.sqrt(var_X_pos)
        else:
            return mu_m_pos

    def sample_y(self, X: NDArray[np.int_], n_samples: int) -> NDArray:
        """Sample from posterior of sparse GP.

        Parameters
        ----------
        X : NDArray[np.int_]
            Indices of features to extract for kernel.
            
        n_samples : int
            Number of times to sample each entry in `X` from the posterior.

        Returns
        -------
        NDArray
            posterior samples.
        """
        prior_mu = self.y_train.mean()
        sig_xm_query, sig_mm_query, sig_mm_pos, mu_m_pos = self._perform_sparse_manipulations(X, prior_mu)

        samples_M_pos = np.random.multivariate_normal(mu_m_pos, sig_mm_pos, n_samples).T
        samples_X_pos = prior_mu + np.matmul(sig_xm_query, np.linalg.solve(sig_mm_query, samples_M_pos - prior_mu))
        
        return samples_X_pos


# ---------------------------------------------------------------------------------------------------------------


# class SparseMKL(SparseGaussianProcess):

#     def __init__(self, kernels: List[Kernel], X_inducing: NDArray[NDArray[np.float_]], jitter: float = 0.000001) -> None:
#         super().__init__(None, X_inducing, jitter)  # pass model as none, bad code but does the job since set later on
#         self.kernels = kernels
#         nk = len(self.kernels)
#         self.weights = np.ones(nk) / nk  # start off with evenly weighted kernels
#         self._sampled_weights = [self.weights]
#         self._weight_space = np.random.RandomState(1).dirichlet(np.ones(nk), size=1_000)
#         self._reward_values = []
#         self._optimmiser = GaussianProcessRegressor(kernel=RBF(lengthscales=np.ones(self.weight_space.shape[1])))

#     @staticmethod
#     def _acquisition(mu, sigma, y_max):
#         # expected improvement sampling
#         improvement = mu - y_max
#         scaled_mu = np.divide(improvement, sigma)
#         alpha = improvement * norm.cdf(scaled_mu) + sigma * norm.pdf(scaled_mu)
#         ranked = np.argsort(alpha)
#         return ranked

#     def get_weighted_kernel(self):
#         return sum([w * k for w, k in zip(self.weights, self.kernels)] + [self.white_noise_kernel])  # add it here but dont weight

#     def fit(self, X_train, y_train):
#         assert False, ' this is nice but wont work since cant handle multiple fingerprints unless the tanimoto kernel also contains the HDF5 dataset which is loaded from...'
#         self.model = GaussianProcessRegressor(kernel=self.get_weighted_kernel())
#         super().fit(X_train, y_train)
#         y_pred, _ = self.model.predict(X_train)
#         self._reward_values.append(-np.median(abs(y_train - y_pred)))  # median absolute error
        
#         self._optimiser.fit(self._sampled_weights, self._reward_values)
#         mu, std = self._optimmiser.predict(self._weight_space)    
#         maximises_score = self._acquisition(mu, std, max(self._reward_values))
#         self.weights = self._weight_space[maximises_score[-1]]
        
#         self._sampled_weights.append(self.weights)
#         self.model = GaussianProcessRegressor(kernel=self.get_weighted_kernel())


# # ---------------------------------------------------------------------------------------------------------------


class EnsembleSparseGaussianProcess:

    def __init__(self, rbf_model: SparseGaussianProcess, mkl_model: SparseMKL) -> None:
        self.rbf_model = rbf_model
        self.mkl_model = mkl_model

    def fit(self, X_train: NDArray[np.int_], y_train: NDArray[np.float_]) -> None:
        """fit models in ensemble to passed data.

        Parameters
        ----------
        X_train : NDArray[np.int_]
            Indices of features to extract for kernel.
            
        y_train : NDArray[np.float_]
            performance values (not indices) of the passed MOF indices.
        """
        self.y_train = np.array(y_train)

        self.rbf_model.fit(X_train, y_train)
        self.mkl_model.fit(X_train, y_train)

    def predict(self, X: NDArray[np.int_], return_std: bool=False) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
        """Make predictions for passed MOFs

        Parameters
        ----------
        X : NDArray[np.int_]
            Indices of features to extract for kernel.
            
        return_std : bool, optional
            return the std of the model predictions (not calculated if value is False)

        Returns
        -------
        Union[Tuple[NDArray, NDArray], NDArray]
            Either two 1D arrays or a single array depending if `return_std` is specified.
        """
        def _calc_precision(std):
            return 1 / (std ** 2)

        mu_rbf, std_rbf = self.rbf_model.predict(X, return_std=True)
        mu_mkl, std_mkl = self.mkl_model.predict(X, return_std=True)
        
        p1, p2 = _calc_precision(std_rbf), _calc_precision(std_mkl)
        p = p1 + p2
        
        mu = ((p1 * mu_rbf) + (p2 * mu_mkl)) / p
        
        if return_std:
            std = 1 / np.sqrt(p)
            return mu, std
        else:
            return mu

    def sample_y(self, X: NDArray[np.int_], n_samples: int) -> Union[NDArray[np.float_], NDArray[NDArray[np.float_]]]:
        """Sample from posterior of sparse GP.

        Parameters
        ----------
        X : NDArray[np.int_]
            Indices of features to extract for kernel.
            
        n_samples : int
            Number of times to sample each entry in `X` from the posterior.

        Returns
        -------
        NDArray
            posterior samples.
        """
        post_rbf = self.rbf_model.sample_y(X, n_samples)
        post_mkl = self.mkl_model.sample_y(X, n_samples)        
        return np.hstack((post_rbf, post_mkl))

    def get_max_y(self) -> np.float_:
        return self.y_train.max()


# # ---------------------------------------------------------------------------------------------------------------
# TODO : 




# # TODO : write the inducing functions (cluster then return centroids), consider how to do this for PCFP
## >* the centroids to be the result of performing a clustering on each dataset and selectingn/4 centroids from each, final matrix to be the combination of all 4 centroids