from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray

from mkl.sparse import EnsembleSparseGaussianProcess
from mkl.acquisition import GreedyNRanking


class SurrogateModel:
    
    def __init__(self, model, acquisitor) -> None:
        """
        Parameters
        ----------
        model : model object
            Must have `fit(X_ind, y)` and `sample_y(n_samples)` methods.            
        """
        self.model = model
        self.acquisitor = acquisitor
        
    def fit(self, X_ind: NDArray[np.int_], y_val: NDArray[np.int_]) -> None:
        """Fit model to passed data points

        Parameters
        ----------
        X_ind : NDArray[np.int_]
            indices of data points to use.
            
        y_val : NDArray[np.int_]
            taret values of data points.
        """
        self.model.fit(X_ind, y_val)        
        
    def rank(self, X_ind: NDArray[np.int_]) -> NDArray[np.float_]:
        """Rank the passed indices from highest to lowest.
        Highest ranked are highest recommended to be sampled.

        Parameters
        ----------
        X_ind : NDArray[np.int_]
            _description_

        Returns
        -------
        NDArray[np.float_]
            ranked highest to lowest, element 0 is largest ranked, element -1 is lowest ranked.
        """
        alpha = self.determine_alpha(X_ind)
        return alpha  # index of largest alpha is first
    
    @abstractmethod
    def determine_alpha(X_ind: NDArray[np.int_]) -> NDArray[np.float_]:
        """Abstract method to define interaction betwee the model and acquisition

        Parameters
        ----------
        X_ind : NDArray[np.int_]
            _description_

        Returns
        -------
        NDArray[np.float_]
            _description_
        """
        return NotImplemented



class SparseSeparationSurrogate(SurrogateModel):
    """Sparse surroagte model for a separation process using a posterior sampling based acquisition method.
    As selecting points to sample for a separation process, will use the absolute of the posterior samples to rank
    the data points as the logorathm will be used instead of absolute values.
    """
    
    def __init__(self, model: EnsembleSparseGaussianProcess, acquisitor=GreedyNRanking, n_post: int=50) -> None:
        super().__init__(model, acquisitor)
        self.n_post = int(n_post)
            
    def determine_alpha(self, X_ind: NDArray[np.int_]) -> NDArray[np.float_]:
        """Determine the alpha (ranking values) for each data point in `X_ind`.
        Performs 50 posterior samples for each instance in self.model and determines the absolute values.

        Parameters
        ----------
        X_ind : NDArray[np.int_]
            indices of data points to use.
            Sparse process will sample on entire dataset so sample then index afterwards.

        Returns
        -------
        NDArray[np.float_]
            ranked alpha terms for the specified indices ranked highest t lowest where highest is most recommended.
        """
        posterior = abs(self.model.sample_y(n_samples=self.n_post))[X_ind]
        alpha = self.acquisitor.score_points(posterior)
        alpha = np.argsort(alpha)[::-1]
        return alpha