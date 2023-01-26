import numpy as np
from numpy.typing import NDArray

from mkl.model import EnsembleSparseGaussianProcess
from mkl.acquisition import GreedyNRanking


class GreedySurrogateModel:
    
    def __init__(self, model: EnsembleSparseGaussianProcess, n_post: int, n_opt: int) -> None:
        self.model = model
        self.n_post = int(n_post)
        self.acquisitor = GreedyNRanking(n_opt=n_opt)
    
    def fit(self, x: NDArray[np.int_], y: NDArray[np.int_]) -> None:
        self.model.fit(x, y)
        
    def rank(self, x: NDArray[np.int_]) -> NDArray[np.int_]:
        alpha = self.determine_alpha(x)
        return np.argsort(alpha)[::-1]  # index of largest alpha is first
            
    def determine_alpha(self, x: NDArray[np.int_]) -> NDArray[np.float_]:
        posterior = abs(self.model.sample_y(x, n_samples=self.n_post))
        alpha = self.acquisitor.score_points(posterior)
        return alpha