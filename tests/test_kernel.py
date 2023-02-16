import numpy as np
import pytest
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from mkl.kernel import TanimotoKernel

# -----------------------------------------------------------------------------------------------------------------------------

X_FP = np.array([
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 1],
    [1, 0, 0, 0],
    [0, 1, 1, 1],
]).astype(int)    

TAN_FP = np.array(
    [
        [1.  , 1.  , 0.5 , 0.5 , 0.25],
        [1.  , 1.  , 0.5 , 0.5 , 0.25],
        [0.5 , 0.5 , 1.  , 0.25, 0.75],
        [0.5 , 0.5 , 0.25, 1.  , 0.  ],
        [0.25, 0.25, 0.75, 0.  , 1.  ]
        ]
    )




# -----------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("var", 
                         [
                             1.0,
                             2.0
                         ])
def test_TanimotoKernel(var):
    # proves get correct output and that fingerprints must be floats instead of ints
    kernel = TanimotoKernel(variance=var)
    X = X_FP.astype(float)
    
    k_x = kernel.K(X).numpy()
    k_xx = kernel.K(X, X).numpy()
    
    assert np.array_equal(k_x, k_xx)
    assert np.array_equal(TAN_FP * var, k_x)
    
    X = X_FP.astype(int)
    with pytest.raises(InvalidArgumentError):
        k_x = kernel.K(X)
    
    















# -----------------------------------------------------------------------------------------------------------------------------
