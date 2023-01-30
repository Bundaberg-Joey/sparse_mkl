from typing import Union, Tuple

import numpy as np
from numpy.typing import NDArray
import GPy


class SparseGaussianProcessGPy:
    """Gaussian process capabale of performing sparse manipulations to speed up predictions / posterior samplng on large datasets.
    """

    def __init__(self, X_inducing: NDArray[np.int_], jitter: float=1e-6) -> None:
        """
        Parameters
        ----------
        X_inducing : NDArray[np.int_]
            Indices of MOFs to use for inducing matrix.
            
        jitter : float
            slight noise value to prevent numerical issues.
        """
        self.X_inducing = np.array(X_inducing)  # indices of MOFs in inducing array 
        self.jitter = float(jitter)
        self.kmm = None
        self.internal_model = None

    def fit(self, X_train: NDArray[np.int_], y_train: NDArray[np.float_]) -> None:
        """Fit model to passed data.

        Parameters
        ----------
        X_train : NDArray[np.int_]
            Indices of features to extract for kernel.
            
        y_train : NDArray[np.float_]
            performance values (not indices) of the passed MOF indices.
        """        
        self.X_inducing = np.vstack((self.X_inducing, X_train))  # update with training points
        self.y_train = y_train
        n_feat = X_train.shape[1]
        
        mfy = GPy.mappings.Constant(input_dim=n_feat, output_dim=1)  # fit dense GPy model to this data
        ky = GPy.kern.RBF(n_feat, ARD=True, lengthscale=np.ones(n_feat))
        self.internal_model = GPy.models.GPRegression(X_train, y_train.reshape(-1, 1), kernel=ky, mean_function=mfy)
        self.internal_model.optimize('bfgs')
        
        self.prior_mu = float(self.internal_model.constmap.C)
        self.kernel_var = float(self.internal_model.kern.variance)
        self.prior_var = float(self.internal_model.Gaussian_noise.variance)
        
        k_xm = self.internal_model.kern.K(X_train, self.X_inducing)
        self.k_mm = self.internal_model.kern.K(self.X_inducing, self.X_inducing)
                                
        self.sig_xm_train = self.kernel_var * k_xm
        self.sig_mm_train = (self.kernel_var * self.k_mm )+ np.identity(n=len(self.X_inducing)) * self.kernel_var * self.jitter
        self.updated_var = self.kernel_var + self.prior_var - np.sum(np.multiply(np.linalg.solve(self.sig_mm_train, self.sig_xm_train.T), self.sig_xm_train.T), 0)
        
    def _perform_sparse_manipulations(self, X: NDArray[np.float_], prior_mu: float) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """relevant matrix transformations / operations to achieve sparse GP

        Parameters
        ----------
        X : NDArray[np.float]
            training features (explicit)
            
        prior_mu : float
            mean of he prior distribution trget values.

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray, NDArray]
            transformed matrices.
        """

        k_xm_query = self.internal_model.kern.K(X, self.X_inducing)
        k_mm_query = self.k_mm

        sig_xm_query = self.kernel_var * k_xm_query
        sig_mm_query = self.kernel_var * k_mm_query + np.identity(len(self.X_inducing)) * self.jitter * self.kernel_var

        K = np.matmul(self.sig_xm_train.T, np.divide(self.sig_xm_train, self.updated_var.reshape(-1, 1)))
        sig_mm_pos = sig_mm_query - K + np.matmul(K, np.linalg.solve(K + sig_mm_query, K))
        J = np.matmul(self.sig_xm_train.T, np.divide(self.y_train - prior_mu, self.updated_var))
        mu_m_pos = prior_mu + J - np.matmul(K, np.linalg.solve(K + self.sig_mm_train, J))
        return  sig_xm_query, sig_mm_query, sig_mm_pos, mu_m_pos

    def predict(self, X: NDArray[np.float_], return_std: bool=False) -> Union[Tuple[NDArray, NDArray], NDArray]:
        """Make predictions for passed MOFs

        Parameters
        ----------
        X : features (explicit)
            
        return_std : bool, optional
            return the std of the model predictions (not calculated if value is False)

        Returns
        -------
        Union[Tuple[NDArray, NDArray], NDArray]
            Either two 1D arrays or a single array depending if `return_std` is specified.
        """
        sig_xm_query, sig_mm_query, sig_mm_pos, mu_m_pos = self._perform_sparse_manipulations(X, self.prior_mu)
       
        mu_X_pos = self.prior_mu + np.matmul(sig_xm_query, np.linalg.solve(sig_mm_query, mu_m_pos - self.prior_mu))

        if return_std:
            var_X_pos = np.sum(np.multiply(np.matmul(np.linalg.solve(sig_mm_query, np.linalg.solve(sig_mm_query, sig_mm_pos).T), sig_xm_query.T), sig_xm_query.T), 0)
            return mu_X_pos, np.sqrt(var_X_pos)
        else:
            return mu_X_pos

    def sample_y(self, X: NDArray[np.float_], n_samples: int) -> NDArray:
        """Sample from posterior of sparse GP.

        Parameters
        ----------
        X : NDArray[np.foat]
            Features (explicit)
            
        n_samples : int
            Number of times to sample each entry in `X` from the posterior.

        Returns
        -------
        NDArray
            posterior samples.
        """
        sig_xm_query, sig_mm_query, sig_mm_pos, mu_m_pos = self._perform_sparse_manipulations(X, self.prior_mu)

        samples_M_pos = np.random.multivariate_normal(mu_m_pos, sig_mm_pos, n_samples).T
        samples_X_pos = self.prior_mu + np.matmul(sig_xm_query, np.linalg.solve(sig_mm_query, samples_M_pos - self.prior_mu))
        
        return samples_X_pos


#-----------------------------------------------------------------------------------------------------------------
