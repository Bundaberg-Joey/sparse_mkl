from typing import Tuple, List
import numpy as np
from numpy.typing import NDArray

import gpflow

from mkl.kernel import TanimotoKernel


# ------------------------------------------------------------------------------------------------------------------------------------

class DenseGpflowModel:
    
    def __init__(self, X, X_M):
        self.X = X  # data object
        self.M = self.X[X_M]  # indices passed to data object (M is a data object)
    
    def fit(self, X_ind, y_val):
        self.M = np.vstack((self.X[X_ind], self.M))  # cotinually incorporates training data
        
        x_ = self.X[X_ind]
        y_ = np.reshape(y_val, (-1, 1))
        
        self.model = self.build_model(x_, y_)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(self.model.training_loss, self.model.trainable_variables)  
    
    def get_prior_mu(self):
        return self.model.mean_function.c.numpy()

    def get_kernel_var(self):
        return self.model.kernel.variance.numpy()
    
    def get_gaussian_var(self):
        return self.model.likelihood.variance.numpy()
    
    def calc_k_xm(self):
        return self.model.kernel.K(self.X, self.M).numpy()
    
    def calc_k_mm(self):
        return self.model.kernel.K(self.M, self.M).numpy()
    


class DenseRBFModel(DenseGpflowModel):

    @staticmethod
    def build_model(X, y):
        model = gpflow.models.GPR(
        data=(X, y), 
        kernel=gpflow.kernels.RBF(lengthscales=np.ones(X.shape[1])),
        mean_function=gpflow.mean_functions.Constant()
        )
        return model
        
        
class DenseTanimotoModel(DenseGpflowModel):
    
    @staticmethod
    def build_model(X, y):
        model = gpflow.models.GPR(
            data=(X, y),
            kernel=TanimotoKernel(),
            mean_function=gpflow.mean_functions.Constant()
        )
        return model
    
    
class DenseMultipleKernelLearner(DenseTanimotoModel):
    
    def __init__(self, X: List, X_M: List):  # each can have their own inducing matrix!!
        self.X = X  #list of data objects
        self.M = [xi[xm] for xi, xm in zip(self.X, X_M)]  # X_m list of indices (one per kernel) self.M is list of data objects
        self.n_kernels = len(self.X)
        self.weights = np.ones(self.n_kernels) / self.n_kernels  # equal weights
        self.models = []
        
    def fit(self, X_ind, y_val):
        
        self.models = []
                
        for i in range(self.n_kernels):
            self.M[i] = np.vstack((self.X[i][X_ind], self.M[i]))  # update each inducing matrix with training indices
            
            x_ = self.X[i][X_ind]
            y_ = np.reshape(y_val, (-1, 1))
            model = self.build_model(x_, y_)
            
            opt = gpflow.optimizers.Scipy()
            opt.minimize(model.training_loss, model.trainable_variables)  
            self.models.append(model)
                        
    def get_prior_mu(self):
        return sum([w * m.mean_function.c.numpy() for w, m in zip(self.weights, self.models)])

    def get_kernel_var(self):
        return sum([w * m.kernel.variance.numpy() for w, m in zip(self.weights, self.models)])
    
    def get_gaussian_var(self):
        return sum([w * m.likelihood.variance.numpy() for w, m in zip(self.weights, self.models)])

    def calc_k_xm(self):
        k = self._calc_weighted_k(self.X, self.M)
        print(F'k_xm={k.shape}')
        return k
            
    def calc_k_mm(self):
        k = self._calc_weighted_k(self.M, self.M)
        print(F'k_mm={k.shape}')
        return k
            

    def _calc_weighted_k(self, A, B):
        k = []
        
        for i in range(self.n_kernels):
            k_ = self.models[i].kernel.K(A[i], B[i]).numpy()
            k.append(k_ * self.weights[i])
            
        return sum(k)

      

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

    
class Ensemble:
    
    def __init__(self, sparse_rbf, sparse_mkl) -> None:
        self.sparse_rbf = sparse_rbf
        self.sparse_mkl = sparse_mkl
        self.update_counter = 0
        self.updates_per_big_fit = 10
        self.ntop=100 
        self.nmax=400
    
    def fit(self, X_train, y_train):
        # same update logic as single sparse model but allows for ensemble to keep track / apply methods independently
        
        if self.update_counter % self.updates_per_big_fit == 0:
            
            n_tested = len(y_train)
            # if more than nmax we will subsample and use the subsample to fit hyperparametesr
            if n_tested <= self.nmax:
                pass

            else:
                # subsample if above certain number of points to keep "fitting" fast
                top_ind = np.argsort(y_train)[-self.ntop:]  # indices of top y sampled so far
                rand_ind = np.random.choice([i for i in range(n_tested) if i not in top_ind], replace=False, size=n_tested-self.ntop)  # other indices
                chosen = np.hstack((top_ind, rand_ind))
                y_train = y_train[chosen]
                X_train = X_train[chosen]

    
            self.sparse_rbf.update_params(X_train, y_train)
            self.sparse_mkl.update_params(X_train, y_train)
        
        self.sparse_rbf.update_data(X_train, y_train)
        self.sparse_mkl.update_data(X_train, y_train)
        self.update_counter += 1
    
    def predict(self):
        def _calc_precision(std):
            return 1 / (std ** 2)

        mu_rbf, std_rbf = self.sparse_rbf.predict()
        mu_mkl, std_mkl = self.sparse_mkl.predict()
        
        p1, p2 = _calc_precision(std_rbf), _calc_precision(std_mkl)
        p = p1 + p2
        
        mu = ((p1 * mu_rbf) + (p2 * mu_mkl)) / p
        std = 1 / np.sqrt(p)
        return mu, std
    
    def sample_y(self, n_samples):
        post_rbf = self.sparse_rbf.sample_y(n_samples)
        post_mkl = self.sparse_mkl.sample_y(n_samples)   
        # likely a better way to combine than just a stacking approach?     
        return np.hstack((post_rbf, post_mkl))


# ------------------------------------------------------------------------------------------------------------------------------------
