from typing import List, Union, Optional
from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray
import gpflow

from mkl.kernel import TanimotoKernel
from mkl.data import Hdf5Dataset


# ------------------------------------------------------------------------------------------------------------------------------------

class DenseGpflowModel:
    """Dense model is fit to all of training data and used to extract relevant hyperparameters for the sparse model.
    Uses `gpflow` as a backend.
    """
    
    def __init__(self, X: Union[NDArray, Hdf5Dataset], M: NDArray[NDArray[np.float_]]) -> None:
        """
        Parameters
        ----------
        X : Union[NDArray, Hdf5Dataset]
            Either 2d numpy array  or HDF5 dataset arranged with rows as entries and columns as features.
            
        M : NDArray[NDArray[np.float_]]
            A 2d numpy array to use as inducing points for sparse process

        Returns
        -------
        None
        """
        self.X = X
        self.M = M
        
    @abstractmethod
    def build_model(self, X: NDArray[NDArray[np.float_]], y: NDArray[np.float_]) -> gpflow.models.GPR:
        """Initialise and return the gpflow model (will be optimised when `fit` is called).
        Just a covenience method for subclassing to create different dense models more flexibly.

        Parameters
        ----------
        X : NDArray[NDArray[np.float_]]
            Feature matrix to fit model to, rows are entries and columns are features.
            
        y : NDArray[np.float_]
            Target values for passed entries.

        Returns
        -------
        gpflow.models.GPR
        """
        return NotImplemented
        
    def fit(self, X_ind: NDArray[np.int_], y_val: NDArray[np.float_]) -> None:
        """Fit the backend gpflow model to the passed data.

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
        self.M = np.vstack((self.X[X_ind], self.M))  # cotinually incorporates training data
        
        x_ = self.X[X_ind]
        y_ = np.reshape(y_val, (-1, 1))
        
        self.model = self.build_model(x_, y_)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(self.model.training_loss, self.model.trainable_variables)  
    
    def get_prior_mu(self) -> np.float_:
        """Get the mean of the prior data as determined by gpflow model.

        Returns
        -------
        np.float_
        """
        return self.model.mean_function.c.numpy()

    def get_kernel_var(self) -> np.float_:
        """Get the variance of the kernel as determined by gpflow model.

        Returns
        -------
        np.float_
        """
        return self.model.kernel.variance.numpy()
    
    def get_gaussian_var(self) -> np.float:
        """Get the variance of the gaussian noise as determined by gpflow model.

        Returns
        -------
        np.float_
        """
        return self.model.likelihood.variance.numpy()
    
    def calc_k_xm(self) -> NDArray[NDArray[np.float_]]:
        """Calculates the covariance matrix between each entry in the fill dataset `X` and the inducing matrix `X_M`.
        Calcualtion uses fitted gpflow model.

        Returns
        -------
        NDArray[NDArray[np.float_]]
        """
        return self.model.kernel.K(self.X[:], self.M[:]).numpy()
    
    def calc_k_mm(self) -> NDArray[NDArray[np.float_]]:
        """Calculates the covariance matrix between each entry in the inducing matrix against itself.
        Calcualtion uses fitted gpflow model.

        Returns
        -------
        NDArray[NDArray[np.float_]]
        """
        return self.model.kernel.K(self.M[:], self.M[:]).numpy()
    

# ------------------------------------------------------------------------------------------------------------------------------------


class DenseRBFModel(DenseGpflowModel):
    """Dense gpflow model which uses a constant mean function and an RBF kernel.
    Overloads `build_model` to achieve this.
    """
    
    def build_model(self, X: NDArray[NDArray[np.float_]], y: NDArray[np.float_]) -> gpflow.models.GPR:
        # see docstring in `DenseGpflowModel`
        model = gpflow.models.GPR(
        data=(X, y), 
        kernel=gpflow.kernels.RBF(lengthscales=np.ones(X.shape[1])),
        mean_function=gpflow.mean_functions.Constant()
        )
        return model
        
            
# ------------------------------------------------------------------------------------------------------------------------------------
    
        
class DenseTanimotoModel(DenseGpflowModel):
    """Dense gpflow model which uses a constant mean function and a Tanimoto / Jaccard kernel.
    Overloads `build_model` to achieve this.
    """

    def build_model(self, X: NDArray[NDArray[np.float_]], y: NDArray[np.float_]) -> gpflow.models.GPR:
        # see docstring in `DenseGpflowModel`
        model = gpflow.models.GPR(
            data=(X, y),
            kernel=TanimotoKernel(),
            mean_function=gpflow.mean_functions.Constant()
        )
        return model
    
    
# ------------------------------------------------------------------------------------------------------------------------------------

    
class DenseMultipleKernelLearner(DenseTanimotoModel):
    """A dense gpflow model which supports weighted kernels within a single model.
    Because `gpflow` doesnt natively support weighted kernel combinations individual internal models are created and used.
    
    The kernel weights in this model are passed at init and remain fixed.
    For a dense model with dynamic weight adjustment see `DynamicDenseMKL`.
    """
    
    def __init__(self, X: List[Union[NDArray, Hdf5Dataset]], M: List[NDArray[NDArray[np.float_]]], weights: Optional[NDArray[np.float_]]=None) -> None:  # each can have their own inducing matrix!!
        """
        Parameters
        ----------
        X : List[Union[NDArray, Hdf5Dataset]]
            List of dataset objects / 2d numpy arrays, each kernel will have its own dataset object.
            Allows for different features to be used per kernel.
            
        M: List[NDArray[NDArray[np.float_]]]
            List of inducing point arrays to be used per kernel.
            Allows for different kernels supporting different features to be used as feature landscape will differ in each case.
            
        weights : Optional[NDArray[np.float_]], optional
            scaler weights to use for each kernel.
            If not passed then equal weighting will be used for each kernel
            Total weights must sum to 1.0, passed weights will be normalised to facilitate this.
        """
        self.X = X 
        self.M = M
        self.n_kernels = len(self.X)
        self.models = []
        
        weights = np.ones(self.n_kernels) / self.n_kernels if weights is None else np.array(weights)
        self.weights = weights / weights.sum()  # normalise to achieve bounding between zero and one.

        
    def fit(self, X_ind: NDArray[np.int_], y_val: NDArray[np.float_]) -> None:
        """Fit the backend gpflow models to the passed data.

        Parameters
        ----------
        X_ind : NDArray[np.int_]
            Indices of data points to use when fitting ech model
            They are also incorporated into the inducing feature matrix for each model each time fit is called.
            
        y_val : NDArray[np.float_]
            Target values for each entry. 
            
        Returns
        -------
        None
        """
        self.models = []
                
        for i in range(self.n_kernels):
            self.M[i] = np.vstack((self.X[i][X_ind], self.M[i][:]))  # update each inducing matrix with training indices
            
            x_ = self.X[i][X_ind]
            y_ = np.reshape(y_val, (-1, 1))
            model = self.build_model(x_, y_)
            
            opt = gpflow.optimizers.Scipy()
            opt.minimize(model.training_loss, model.trainable_variables)  
            self.models.append(model)
                        
    def get_prior_mu(self) -> np.float_:
        """Get the weighted mean of the prior data as determined by gpflow model.
        Returned as the weighted sum from each model.

        Returns
        -------
        np.float_
        """
        return sum([w * m.mean_function.c.numpy() for w, m in zip(self.weights, self.models)])

    def get_kernel_var(self) -> np.float_:
        """Get the weighted variance of the kernel as determined by gpflow model.
        Returned as the weighted sum from each model.
        
        Returns
        -------
        np.float_
        """
        return sum([w * m.kernel.variance.numpy() for w, m in zip(self.weights, self.models)])
    
    def get_gaussian_var(self) -> np.float:
        """Get the variance of the gaussian noise as determined by gpflow model.
        Returned as the weighted sum from each model.
        
        Returns
        -------
        np.float_
        """
        return sum([w * m.likelihood.variance.numpy() for w, m in zip(self.weights, self.models)])

    def calc_k_xm(self) -> NDArray[NDArray[np.float_]]:
        """Calculates the weighted covariance matrix between each entry in the fill dataset `X` and the inducing matrix `X_M`.
        Calcualtion uses fitted gpflow models.

        Returns
        -------
        NDArray[NDArray[np.float_]]
        """
        k = self._calc_weighted_k(self.X, self.M)
        return k
            
    def calc_k_mm(self) -> NDArray[NDArray[np.float_]]:
        """Calculates the weighted covariance matrix between each entry in the inducing matrix against itself.
        Calcualtion uses fitted gpflow models.

        Returns
        -------
        NDArray[NDArray[np.float_]]
        """
        k = self._calc_weighted_k(self.M, self.M)
        return k
            

    def _calc_weighted_k(self, A: List[Union[NDArray, Hdf5Dataset]], B: List[Union[NDArray, Hdf5Dataset]]) -> NDArray[NDArray[np.float_]]:
        """Calcualte the weighted covariance matrix for the passed lists of feature matrices.

        Parameters
        ----------
        A : List[Union[NDArray, Hdf5Dataset]]
            Feature matrix `A`.
            
        B : List[Union[NDArray, Hdf5Dataset]]
            Feature matrix `A`.

        Returns
        -------
        NDArray[NDArray[np.float_]]
            summation of weighted covariance matrices.
        """
        k = []
        
        for i in range(self.n_kernels):
            k_ = self.models[i].kernel.K(A[i][:], B[i][:]).numpy()
            k.append(k_ * self.weights[i])
            
        return sum(k)


# ------------------------------------------------------------------------------------------------------------------------------------


class DynamicDenseMKL(DenseMultipleKernelLearner):
    """A dense gpflow model which supports weighted kernels within a single model.
    Because `gpflow` doesnt natively support weighted kernel combinations individual internal models are created and used.
    
    The kernel weights within this model are updated each time `fit` is called using the kernel allignment method.
    """

    @staticmethod
    def calc_frobenius_product(Ka: NDArray[NDArray[np.float_]], Kb: NDArray[NDArray[np.float_]]) -> float:
        """calculate frobenius product between kernel matrices
        Parameters
        ----------
        Ka : NDArray[NDArray[np.float_]]
            
        Kb : NDArray[NDArray[np.float_]]
        
        Returns
        -------
        float
            allignment score with `1.0` being the highest allignment score and lowest being `0.0`
        """
        return np.einsum('ij,ij->', Ka, Kb)
    
    def fit(self, X_ind: NDArray[np.int_], y_val: NDArray[np.float_]) -> None:
        """Fit the backend gpflow models to the passed data.
        The kernel weights are also updated during this call to `fit`.

        Parameters
        ----------
        X_ind : NDArray[np.int_]
            Indices of data points to use when fitting ech model
            They are also incorporated into the inducing feature matrix for each model each time fit is called.
            
        y_val : NDArray[np.float_]
            Target values for each entry. 
            
        Returns
        -------
        None
        """
        super().fit(X_ind, y_val)
        
        # re-weight kernels using "kernel allignment method"
        y_val = y_val.reshape(-1, 1)
        yc = ((y_val - y_val.mean()) / y_val.std())
        Kyy = np.dot(yc, yc.T)
        bb = self.calc_frobenius_product(Kyy, Kyy)
        
        allignments = np.ones(self.n_kernels)
        
        for i in range(self.n_kernels):
            ka = self.models[i].kernel.K(self.X[i][X_ind]).numpy()
            aa, ab = (self.calc_frobenius_product(x, y) for x, y in ((ka, ka), (ka, Kyy)))
            allignments[i] = ab / (np.sqrt(aa * bb))
        
        self.weights = allignments / np.sum(allignments)
        
    

# ------------------------------------------------------------------------------------------------------------------------------------
