from typing import List
import numpy as np

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
    

# ------------------------------------------------------------------------------------------------------------------------------------


class DenseRBFModel(DenseGpflowModel):

    @staticmethod
    def build_model(X, y):
        model = gpflow.models.GPR(
        data=(X, y), 
        kernel=gpflow.kernels.RBF(lengthscales=np.ones(X.shape[1])),
        mean_function=gpflow.mean_functions.Constant()
        )
        return model
        
    
# ------------------------------------------------------------------------------------------------------------------------------------
    
        
class DenseTanimotoModel(DenseGpflowModel):
    
    @staticmethod
    def build_model(X, y):
        model = gpflow.models.GPR(
            data=(X, y),
            kernel=TanimotoKernel(),
            mean_function=gpflow.mean_functions.Constant()
        )
        return model
    
    
# ------------------------------------------------------------------------------------------------------------------------------------

    
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
