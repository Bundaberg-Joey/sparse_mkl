import numpy as np
from sklearn.gaussian_process import kernels, GaussianProcessRegressor

from mkl.model import SparseGaussianProcess


X = np.random.randn(100_000, 5)
y = np.random.randn(100_000)
X_ind = X[-100:]

kernel = kernels.RBF(length_scale=np.ones(5)) + kernels.WhiteKernel()
internal_model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
model = SparseGaussianProcess(internal_model, X_inducing=X_ind)

# about 5-8 seconds to fit to 1_000 and sample for 100_000 and inducing matrix of 100
model.fit(X[:1000], y[:1000])
post = model.sample_y(X, n_samples=100)

# crashes after a minute because of memory issues, THE CODE WORKS WOOHOOO !!!!!
#internal_model.fit(X[:1000], y[:1000])  
#post = internal_model.sample_y(X)

print(post.shape)
print(np.isnan(post).any())
print(np.isinf(post).any())

