from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from mkl.dense import DenseGpflowModel

# ------------------------------------------------------------------------------------------------------------------------------------


class SparseGaussianProcess:

    def __init__(self, dense_model: DenseGpflowModel):
        """
        Parameters
        ----------
        dense_model : DenseGpflowModel
            A dense model whih performs fit and prediction to all of pssed data.
            Used to estimate various hyperparameters which are extracted and used by sparse model.
        """
        self.dense_model = dense_model
        self.lam=1e-6
        
    def update_parameters(self, X_train: NDArray[np.int_], y_train: NDArray[np.float_]) -> None:
        """Update parameters of sparse model from passed training data.
        Separate method to `update_data` because of expense incurred when calcualting / fitting here.

        Parameters
        ----------
        X_train : NDArray[np.int_]
            Indices of data points to use when fitting model.
            
        y_train : NDArray[np.float_]
            Performance values of specified data points for fitting model.
        
        Returns: None
            Updates numerous internal parameters
        """
        self.dense_model.fit(X_train, y_train)
        
        self.prior_mu = self.dense_model.get_prior_mu()
        self.kernel_var = self.dense_model.get_kernel_var()
        self.noise_var = self.dense_model.get_gaussian_var()

        k_mm = self.dense_model.calc_k_mm()
        
        self.sig_xm = self.dense_model.calc_k_xm()
        self.sig_mm = k_mm + (np.identity(k_mm.shape[0]) * self.lam * self.kernel_var) 
        self.updated_var = self.kernel_var + self.noise_var - np.sum(np.multiply(np.linalg.solve(self.sig_mm, self.sig_xm.T), self.sig_xm.T),0)
    
    def update_data(self, X_train: NDArray[np.int_], y_train: NDArray[np.float_]) -> None:
        """Update sparse covaraince matrices with passed data points.
        Separate method to `update_parameters` as is cheaper and so can be called more frequently for a "partial fit"

        Parameters
        ----------
        X_train : NDArray[np.int_]
            Indices of data points to use when fitting model.
            
        y_train : NDArray[np.float_]
            Performance values of specified data points for fitting model.
        
        Returns: None
            Updates numerous internal parameters
        """
        K = np.matmul(self.sig_xm[X_train].T, np.divide(self.sig_xm[X_train], self.updated_var[X_train].reshape(-1, 1)))
        self.SIG_MM_pos = self.sig_mm - K + np.matmul(K, np.linalg.solve(K + self.sig_mm, K))
        J = np.matmul(self.sig_xm[X_train].T, np.divide(y_train - self.prior_mu, self.updated_var[X_train]))
        self.mu_M_pos = self.prior_mu + J - np.matmul(K, np.linalg.solve(K + self.sig_mm, J))

    def predict(self) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
        """Returns a prediction for the full dataset held within the dense model.

        Returns
        -------
        Tuple[NDArray[np.float_], NDArray[np.float_]]
            Returns the predicted mean and standard deviation for the entire dataset.
        """
        mu_X_pos = self.prior_mu + np.matmul(self.sig_xm, np.linalg.solve(self.sig_mm, self.mu_M_pos - self.prior_mu))
        var_X_pos = np.sum(np.multiply(np.matmul(np.linalg.solve(self.sig_mm,np.linalg.solve(self.sig_mm,self.SIG_MM_pos).T), self.sig_xm.T), self.sig_xm.T), 0)
        return mu_X_pos, np.sqrt(var_X_pos)

    def sample_y(self, n_samples: int=1) -> NDArray[NDArray[np.float_]]:
        """Samples on inducing points and then uses conditional mean given sample values on full dataset.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to draw from the posterior distribution for each entry in the database `X`

        Returns
        -------
        NDArray[NDArray[np.float_]]
            matrix whose columns are independent samples of the posterior over the full database `X`.
        """
        samples_M_pos = np.random.multivariate_normal(self.mu_M_pos, self.SIG_MM_pos, n_samples).T
        samples_X_pos = self.prior_mu + np.matmul(self.sig_xm, np.linalg.solve(self.sig_mm, samples_M_pos - self.prior_mu))
        return samples_X_pos
    

# ------------------------------------------------------------------------------------------------------------------------------------

    
class EnsembleSparseGaussianProcess:
    """Ensemble model for sparse gaussian process regressors.
    Combines a sparse RBF kernel model and a MKL using tanimoto kernel learners.
    """
    
    def __init__(self, 
                 sparse_rbf: SparseGaussianProcess, 
                 sparse_mkl: SparseGaussianProcess,
                 param_update_freq: int=10, 
                 n_top: int=100,
                 n_max: int=400) -> None:
        """
        Parameters
        ----------
        sparse_rbf : SparseGaussianProcess
            
        sparse_mkl : SparseGaussianProcess
            
        param_update_freq : int, optional
            How many times `fit` must be called for `update_params` to be called for the ensemble models.
            Will be called on the first instance of fit in all cases.
            `update_params` is a more expensive method than `update_data` hence the difference in update frequency.
        
        n_max : int, optional
            Maximum number of data points to be passed to the ensemble models for `update_params` and `update_data`.
            If breached then data points are subsampled to achieve `n_max`.
    
        n_top : int, optional
            Number of top performers to include in the subsampled matrix passed to `update_params` as part of `n_max`.
            Will be `n_top` of the best performing entries in `y` passed to `fit`.
            
        """
        self.sparse_rbf = sparse_rbf
        self.sparse_mkl = sparse_mkl
        
        self.param_update_freq = int(param_update_freq)
        self.n_top=int(n_top) 
        self.n_max=int(n_max)
        
        self._update_counter = 0
    
    def fit(self, X_ind: NDArray[np.int_], y_val: NDArray[np.float_]) -> None:
        """Fit the ensemble models to the passed data points.
        If the number of data points passed is > self.n_max then the data points will be subsampled to `self.n_max`
        to ensure the fitting time doesnt grow significantly at higher numers of data points.
        
        The parameters of the ensemble models are only updated every `self.param_update_freq` times that `fit` is called.
        This ensures that fast predictions / fitting / posterior samples can be achieved during a screening.

        Parameters
        ----------
        X_ind : NDArray[np.int_]
            Indices of data points to use when fitting.
            They are also incorporated into the inducing feature matrix each time fit is called.
            
        y_val : NDArray[np.float_]
            Target values for each entry. 
            
        Returns
        -------
        None
        """
        # same update logic as single sparse model but allows for ensemble to keep track / apply methods independently
        X_ind = np.asarray(X_ind, dtype=int)
        
        if self._update_counter % self.param_update_freq == 0:
            
            n_tested = len(y_val)
            # if more than nmax we will subsample and use the subsample to fit hyperparametesr
            if n_tested <= self.n_max:
                pass

            else:
                # subsample if above certain number of points to keep "fitting" fast
                top_ind = np.argsort(y_val)[-self.n_top:]  # indices of top y sampled so far
                rand_ind = np.random.choice([i for i in range(n_tested) if i not in top_ind], replace=False, size=self.n_max-self.n_top)  # other indices
                chosen = np.hstack((top_ind, rand_ind)).astype(int)
                
                y_val = y_val[chosen]
                X_ind = X_ind[chosen]

            self.sparse_rbf.update_parameters(X_ind, y_val)
            self.sparse_mkl.update_parameters(X_ind, y_val)
        
        self.sparse_rbf.update_data(X_ind, y_val)
        self.sparse_mkl.update_data(X_ind, y_val)
        self._update_counter += 1
    
    def sample_y(self, n_samples: int=1) -> NDArray[NDArray[np.float_]]:
        """Sample from the posterior of the ensemble sparse models.
        

        Parameters
        ----------
        n_samples : int, optional
            specifies how many samples to draw from EACH model for each data point in the dataset.
            So for `n_samples=1` the output will have shape (n, 2) as two models in the ensemble.

        Returns
        -------
        NDArray[NDArray[np.float_]]
            Will have dimensions (n, n_samples * 2) since draw from each model.
        """
        post_rbf = self.sparse_rbf.sample_y(n_samples)
        post_mkl = self.sparse_mkl.sample_y(n_samples)   
        return np.hstack((post_rbf, post_mkl))
    
    def predict(self) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
        """Predict the performance values and standard deviations for the whole dataset for each of the ensmeble models.
        the predicted mean and variance are combined as per: XXX

        Returns
        -------
        Tuple[NDArray[np.float_], NDArray[np.float_]]
            predicted mean and variance.
        """
        def _calc_precision(std):
            return 1 / (std ** 2)

        mu_rbf, std_rbf = self.sparse_rbf.predict()
        mu_mkl, std_mkl = self.sparse_mkl.predict()
        
        p1, p2 = _calc_precision(std_rbf), _calc_precision(std_mkl)
        p = p1 + p2
        
        mu = ((p1 * mu_rbf) + (p2 * mu_mkl)) / p
        std = 1 / np.sqrt(p)
        return mu, std
    


# ------------------------------------------------------------------------------------------------------------------------------------
