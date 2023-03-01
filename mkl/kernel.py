import tensorflow as tf
from gpflow.kernels import Matern12, Matern32, Matern52


# --------------------------------------------------------------------------------------------------------------------


class ImplementSafeEuclidDistance:
    # as per https://github.com/GPflow/GPflow/issues/490
    # introduces small noise term to distance calcultion for very close vectors
    
    def euclid_dist(self, X, X2):
        r2 = self.square_dist(X, X2)
        return tf.sqrt(r2 + 1e-6)
    

class SafeMatern12(ImplementSafeEuclidDistance, Matern12):
    pass


class SafeMatern32(ImplementSafeEuclidDistance, Matern32):
    pass


class SafeMatern52(ImplementSafeEuclidDistance, Matern52):
    pass


# --------------------------------------------------------------------------------------------------------------------
