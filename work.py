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

from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances

class Prospector:

    def __init__(self, X, X_cls):
        """ Initializes by storing all feature values """
        self.X = X
        self.n, self.d = X.shape
        self.X_cls = X_cls
        self.update_counter = 0
        self.updates_per_big_fit = 10
        self.ntop=100 
        self.nmax=400
        self.lam=1e-6

    def fit(self, tested, ytested):
        """
        Fits hyperparameters and inducing points.
        Fit a GPy dense model to get hyperparameters.
        Take subsample for tested data for fitting.

        :param Y: np.array(), experimentally determined values
        :param STATUS: np.array(), keeps track of which materials have been assessed / what experiments conducted
        """
        X = self.X
        # each 10 fits we update the hyperparameters, otherwise we just update the data which is a lot faster
        if np.mod(self.update_counter, self.updates_per_big_fit) == 0:
            print('fitting hyperparameters')
            # how many training points are there
            ntested = len(tested)
            # if more than nmax we will subsample and use the subsample to fit hyperparametesr
            if ntested <= self.nmax:
                train = tested
                ytrain = ytested
            else:
                # subsample if above certain number of points to keep "fitting" fast
                top_ind = np.argsort(ytested)[-self.ntop:]  # indices of top y sampled so far
                rand_ind = np.random.choice([i for i in range(ntested) if i not in top_ind], replace=False, size=ntested-self.ntop)  # other indices
                chosen = np.hstack((top_ind, rand_ind))
                ytrain = np.array([ytested[i] for i in chosen])
                train = [tested[i] for i in chosen]
                
            # use GPy code to fit hyperparameters to minimize NLL on train data
            mfy = GPy.mappings.Constant(input_dim=self.d, output_dim=1)  # fit dense GPy model to this data
            ky = GPy.kern.RBF(self.d, ARD=True, lengthscale=np.ones(self.d))
            self.internal_model = GPy.models.GPRegression(X[train], ytrain.reshape(-1, 1), kernel=ky, mean_function=mfy)
            self.internal_model.optimize('bfgs')

            self.prior_mu = float(self.internal_model.constmap.C)
            self.kernel_var = float(self.internal_model.kern.variance)
            self.noise_var = float(self.internal_model.Gaussian_noise.variance)

            
            # selecting inducing points for sparse inference
            print('selecting inducing points')
            # matrix of inducing points
            self.M = np.vstack((X[train], self.X_cls))
            # dragons...
            # email james.l.hook@gmail.com if this bit goes wrong!
            print('fitting sparse model')

            self.sig_xm = self.internal_model.kern.K(X, self.M)
            self.sig_mm = self.internal_model.kern.K(self.M, self.M) + (np.identity(self.M.shape[0]) * self.lam * self.kernel_var) 
            self.updated_var = self.kernel_var + self.noise_var - np.sum(np.multiply(np.linalg.solve(self.sig_mm, self.sig_xm.T), self.sig_xm.T),0)
        
        K = np.matmul(self.sig_xm[tested].T, np.divide(self.sig_xm[tested], self.updated_var[tested].reshape(-1, 1)))
        self.SIG_MM_pos = self.sig_mm - K + np.matmul(K, np.linalg.solve(K + self.sig_mm, K))
        J = np.matmul(self.sig_xm[tested].T, np.divide(ytested - self.prior_mu, self.updated_var[tested]))
        self.mu_M_pos = self.prior_mu + J - np.matmul(K, np.linalg.solve(K + self.sig_mm, J))
                
        self.update_counter += 1
        """
        key attributes updated by fit

        self.SIG_XM : prior covarience matrix between data and inducing points
        self.SIG_MM : prior covarience matrix at inducing points

        self.SIG_MM_pos : posterior covarience matrix at inducing points
        self.mu_M_pos : posterior mean at inducing points

        """

    def predict(self):
        """
        Get a prediction on full dataset
        just as in MA50263

        :return: mu_X_pos, var_X_pos:
        """

        mu_X_pos = self.prior_mu + np.matmul(self.sig_xm, np.linalg.solve(self.sig_mm, self.mu_M_pos - self.prior_mu))
        var_X_pos = np.sum(np.multiply(np.matmul(np.linalg.solve(self.sig_mm,np.linalg.solve(self.sig_mm,self.SIG_MM_pos).T), self.sig_xm.T), self.sig_xm.T), 0)
        return mu_X_pos, var_X_pos

    def samples(self, nsamples=1):
        """
        sparse sampling method. Samples on inducing points and then uses conditional mean given sample values on full dataset
        :param nsamples: int, Number of samples to draw from the posterior distribution

        :return: samples_X_pos: matrix whose cols are independent samples of the posterior over the full dataset X
        """
        samples_M_pos = np.random.multivariate_normal(self.mu_M_pos, self.SIG_MM_pos, nsamples).T
        samples_X_pos = self.prior_mu + np.matmul(self.sig_xm, np.linalg.solve(self.sig_mm, samples_M_pos - self.prior_mu))
        return samples_X_pos
    
    