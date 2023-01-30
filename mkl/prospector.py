import numpy as np
import GPy


# ------------------------------------------------------------------------------------------------------------------------------------


class Prospector:

    def __init__(self, X, X_cls):
        """ Initializes by storing all feature values """
        self.X = X
        self.n, self.d = X.shape
        self.X_cls = X_cls
        self.update_counter = 0
        self.updates_per_big_fit = 10
        self.ntop=100 
        self.nmax=400
        self.lam=1e-6

    def fit(self, tested, ytested):
        """
        Fits hyperparameters and inducing points.
        Fit a GPy dense model to get hyperparameters.
        Take subsample for tested data for fitting.

        :param Y: np.array(), experimentally determined values
        :param STATUS: np.array(), keeps track of which materials have been assessed / what experiments conducted
        """
        X = self.X
        # each 10 fits we update the hyperparameters, otherwise we just update the data which is a lot faster
        if np.mod(self.update_counter, self.updates_per_big_fit) == 0:
            print('fitting hyperparameters')
            # how many training points are there
            ntested = len(tested)
            # if more than nmax we will subsample and use the subsample to fit hyperparametesr
            if ntested <= self.nmax:
                train = tested
                ytrain = ytested
            else:
                # subsample if above certain number of points to keep "fitting" fast
                top_ind = np.argsort(ytested)[-self.ntop:]  # indices of top y sampled so far
                rand_ind = np.random.choice([i for i in range(ntested) if i not in top_ind], replace=False, size=ntested-self.ntop)  # other indices
                chosen = np.hstack((top_ind, rand_ind))
                ytrain = np.array([ytested[i] for i in chosen])
                train = [tested[i] for i in chosen]
                
            # use GPy code to fit hyperparameters to minimize NLL on train data
            mfy = GPy.mappings.Constant(input_dim=self.d, output_dim=1)  # fit dense GPy model to this data
            ky = GPy.kern.RBF(self.d, ARD=True, lengthscale=np.ones(self.d))
            self.internal_model = GPy.models.GPRegression(X[train], ytrain.reshape(-1, 1), kernel=ky, mean_function=mfy)
            self.internal_model.optimize('bfgs')

            self.prior_mu = float(self.internal_model.constmap.C)
            self.kernel_var = float(self.internal_model.kern.variance)
            self.noise_var = float(self.internal_model.Gaussian_noise.variance)

            
            # selecting inducing points for sparse inference
            print('selecting inducing points')
            # matrix of inducing points
            self.M = np.vstack((X[train], self.X_cls))
            # dragons...
            # email james.l.hook@gmail.com if this bit goes wrong!
            print('fitting sparse model')

            self.sig_xm = self.internal_model.kern.K(X, self.M)
            self.sig_mm = self.internal_model.kern.K(self.M, self.M) + (np.identity(self.M.shape[0]) * self.lam * self.kernel_var) 
            self.updated_var = self.kernel_var + self.noise_var - np.sum(np.multiply(np.linalg.solve(self.sig_mm, self.sig_xm.T), self.sig_xm.T),0)
        
        K = np.matmul(self.sig_xm[tested].T, np.divide(self.sig_xm[tested], self.updated_var[tested].reshape(-1, 1)))
        self.SIG_MM_pos = self.sig_mm - K + np.matmul(K, np.linalg.solve(K + self.sig_mm, K))
        J = np.matmul(self.sig_xm[tested].T, np.divide(ytested - self.prior_mu, self.updated_var[tested]))
        self.mu_M_pos = self.prior_mu + J - np.matmul(K, np.linalg.solve(K + self.sig_mm, J))
                
        self.update_counter += 1
        """
        key attributes updated by fit

        self.SIG_XM : prior covarience matrix between data and inducing points
        self.SIG_MM : prior covarience matrix at inducing points

        self.SIG_MM_pos : posterior covarience matrix at inducing points
        self.mu_M_pos : posterior mean at inducing points

        """

    def predict(self):
        """
        Get a prediction on full dataset
        just as in MA50263

        :return: mu_X_pos, var_X_pos:
        """

        mu_X_pos = self.prior_mu + np.matmul(self.sig_xm, np.linalg.solve(self.sig_mm, self.mu_M_pos - self.prior_mu))
        var_X_pos = np.sum(np.multiply(np.matmul(np.linalg.solve(self.sig_mm,np.linalg.solve(self.sig_mm,self.SIG_MM_pos).T), self.sig_xm.T), self.sig_xm.T), 0)
        return mu_X_pos, np.sqrt(var_X_pos)

    def samples(self, nsamples=1):
        """
        sparse sampling method. Samples on inducing points and then uses conditional mean given sample values on full dataset
        :param nsamples: int, Number of samples to draw from the posterior distribution

        :return: samples_X_pos: matrix whose cols are independent samples of the posterior over the full dataset X
        """
        samples_M_pos = np.random.multivariate_normal(self.mu_M_pos, self.SIG_MM_pos, nsamples).T
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
