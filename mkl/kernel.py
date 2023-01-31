from typing import Optional

import gpflow
from gpflow.utilities import positive, ops
import tensorflow as tf


class TanimotoKernel(gpflow.kernels.Kernel):
    """Tanimoto Kernel implemented using GPflow.
    The variance is used as a kernel parameter and optimised during model optimisation.
    
    This implementation is taken from the below publication:
    
        @article{Thawani2020,
        author = "Aditya Thawani and Ryan-Rhys Griffiths and Arian Jamasb and Anthony Bourached and Penelope Jones and William McCorkindale and Alexander Aldrick and Alpha Lee",
        title = "{The Photoswitch Dataset: A Molecular Machine Learning Benchmark for the Advancement of Synthetic Chemistry}",
        year = "2020",
        month = "7",
        url = "https://chemrxiv.org/articles/preprint/The_Photoswitch_Dataset_A_Molecular_Machine_Learning_Benchmark_for_the_Advancement_of_Synthetic_Chemistry/12609899",
        doi = "10.26434/chemrxiv.12609899.v1"
        }
    """
    def __init__(self):
        super().__init__()
        self.variance = gpflow.Parameter(1.0, transform=positive())

    def K(self, X: tf.Tensor, X2: Optional[tf.Tensor]=None) -> tf.Tensor:
        """Calculates the covaraince matrix using the Tanimoto kernel.
        If `X2` is None then the covariance between each point of `X` withitself is calcualted.
        Else the covaraince between each point in `X` and each point in `X2` is returned.

        Parameters
        ----------
        X : tf.Tensor
            2D tensor containing features (columns) for each data point (rows)
            
        X2 : Optional[tf.Tensor], optional
            Second feature tensor but may be None.

        Returns
        -------
        tf.Tensor
            covariance matrix `K`
        """
        X2 = X if X2 is None else X2
        Xs = tf.reduce_sum(tf.square(X), axis=-1)  # Squared L2-norm of X
        X2s = tf.reduce_sum(tf.square(X2), axis=-1)  # Squared L2-norm of X2
        outer_product = tf.tensordot(X, X2, [[-1], [-1]])  # outer product of the matrices X and X2
        denominator = -outer_product + ops.broadcasting_elementwise(tf.add, Xs, X2s)

        return self.variance * outer_product/denominator

    def K_diag(self, X: tf.Tensor) -> tf.Tensor:
        """The diagonal of the tanimoto kernel is just the variance.
        Hence returns an array of length(X) where each value is the kernel variance.

        Parameters
        ----------
        X : tf.Tensor

        Returns
        -------
        tf.Tensor
        """
        return tf.fill(len(X), self.variance)