from typing import Sequence, Iterator

import numpy as np
from numpy.typing import NDArray

import ami.abc
from ami.abc import SchemaInterface, Feature, Target
from ami.abc.ranker import Index
from ami.schema import Schema

from mkl.sparse import EnsembleSparseGaussianProcess
from mkl.acquisition import GreedyNRanking


class GreedySparseSeparationRanker(ami.abc.RankerInterface):
    def __init__(self, model: EnsembleSparseGaussianProcess, acquisitor: GreedyNRanking, n_post: int=50) -> None:
        """
        Parameters
        ----------
        model : model object
            Must have `fit(X_ind, y)` and `sample_y(n_samples)` methods.            
        """
        self.model = model
        self.acquisitor = acquisitor
        self.n_post = int(n_post)
        
    def fit(self, x: Sequence[Feature], y: Sequence[Target]) -> None:
        """Fit model to passed data points

        Parameters
        ----------
        X_ind : NDArray[np.int_]
            indices of data points to use.
            
        y_val : NDArray[np.int_]
            taret values of data points.
        """
        self.model.fit(x, y)        
        
    def rank(self, x: Sequence[Feature]) -> Iterator[Index]:
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
        alpha = self.determine_alpha(x)
        return alpha  # index of largest alpha is first
                
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
    
    def schema(self) -> SchemaInterface:
        return Schema(
            input_schema=[('index', int)],
            output_schema=[('target', float)]
        )
