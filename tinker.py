__author__ = 'James Hook'
__version__ = '2.0.1'

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets import make_regression

import GPy


class Prospector(object):

    def __init__(self, X):
        """ Initializes by storing all feature values """
        self.X = X
        self.n, self.d = X.shape
        self.update_counter = 0
        self.updates_per_big_fit = 10
        self.estimate_tau_counter = 0
        self.tau_update = 10
        self.y_max = None

    def fit(self, tested, y_train, nkmeans=300, nkeamnsdata=5000,lam=1e-6):
        """
        Fits hyperparameters and inducing points.
        Fit a GPy dense model to get hyperparameters.
        Take subsample for tested data for fitting.

        :param Y: np.array(), experimentally determined values
        :param STATUS: np.array(), keeps track of which materials have been assessed / what experiments conducted
        :param ntop: int, top n samples
        :param nrecent: int, most recent samples
        :param nmax: int, max number of random samples to be taken
        :param ntopmu: int, most promising untested points
        :param ntopvar: int, most uncertain untested points
        :param nkmeans: int, cluster centers from untested data
        :param nkeamnsdata: int, number of sampled points used in kmeans
        :param lam: float, controls jitter in g samples
        """
        X = self.X
        # use GPy code to fit hyperparameters to minimize NLL on train data
        mfy = GPy.mappings.Constant(input_dim=self.d, output_dim=1)  # fit dense GPy model to this data
        ky = GPy.kern.RBF(self.d, ARD=True, lengthscale=np.ones(self.d))
        self.GP = GPy.models.GPRegression(X[train], y_train.reshape(-1, 1), kernel=ky, mean_function=mfy)
        self.GP.optimize('bfgs')
        # strip out fitted hyperparameters from GPy model, because cant do high(ish) dim sparse inference
        self.mu, self.a, self.l, self.b = self.GP.flattened_parameters[:4]
        # selecting inducing points for sparse inference
        # combine with train set above to give nystrom inducing points (inducing points that are also actual trainingdata points)
        nystrom = train
        # also get some inducing points spread throughout domain by using kmeans
        # kmeans is very slow on full dataset so choose random subset
        # also scale using length scales l so that kmeans uses approproate distance measure
        untested = [i for i in range(len(X)) if i not in tested]
        kms = KMeans(n_clusters=nkmeans, max_iter=5).fit(np.divide(X[list(np.random.choice(untested, nkeamnsdata))], self.l))
        # matrix of inducing points
        self.M = np.vstack((X[nystrom], np.multiply(kms.cluster_centers_, self.l)))
        # dragons...
        # email james.l.hook@gmail.com if this bit goes wrong!
        print('fitting sparse model')
        DXM = euclidean_distances(np.divide(X, self.l), np.divide(self.M, self.l), squared=True)
        self.SIG_XM = self.a * np.exp(-DXM / 2)
        DMM = euclidean_distances(np.divide(self.M, self.l), np.divide(self.M, self.l), squared=True)
        self.SIG_MM = self.a * np.exp(-DMM / 2) + np.identity(self.M.shape[0]) * lam * self.a
        self.B = self.a + self.b - np.sum(np.multiply(np.linalg.solve(self.SIG_MM, self.SIG_XM.T), self.SIG_XM.T),0)
        K = np.matmul(self.SIG_XM[tested].T, np.divide(self.SIG_XM[tested], self.B[tested].reshape(-1, 1)))
        self.SIG_MM_pos = self.SIG_MM - K + np.matmul(K, np.linalg.solve(K + self.SIG_MM, K))
        J = np.matmul(self.SIG_XM[tested].T, np.divide(y_train - self.mu, self.B[tested]))
        self.mu_M_pos = self.mu + J - np.matmul(K, np.linalg.solve(K + self.SIG_MM, J))
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

        mu_X_pos = self.mu + np.matmul(self.SIG_XM, np.linalg.solve(self.SIG_MM, self.mu_M_pos - self.mu))
        var_X_pos = np.sum(np.multiply(np.matmul(np.linalg.solve(self.SIG_MM,np.linalg.solve(self.SIG_MM,self.SIG_MM_pos).T), self.SIG_XM.T), self.SIG_XM.T), 0)
        return mu_X_pos, var_X_pos

    def samples(self, nsamples=1):
        """
        sparse sampling method. Samples on inducing points and then uses conditional mean given sample values on full dataset
        :param nsamples: int, Number of samples to draw from the posterior distribution

        :return: samples_X_pos: matrix whose cols are independent samples of the posterior over the full dataset X
        """
        samples_M_pos = np.random.multivariate_normal(self.mu_M_pos, self.SIG_MM_pos, nsamples).T
        samples_X_pos = self.mu + np.matmul(self.SIG_XM, np.linalg.solve(self.SIG_MM, samples_M_pos - self.mu))
        return samples_X_pos



X, y = make_regression(n_samples=30, n_features=5, noise=1.0, random_state=1)
prosp = Prospector(X)

m_train = 10
train = np.arange(m_train)
y_train = y[:m_train]

prosp.fit(train, y_train, nkeamnsdata=22, nkmeans=3)
mu, std = prosp.predict()

print(y_train)
print(y_train.mean())
print(y_train.std())
print(y_train.std() ** 2)
print()
print(prosp.mu)
print(prosp.a)
print(prosp.l)
print(prosp.b)
print()
print(mu)
print()
print(std)






##############################################
# NEED TO INCLUDE THE TRAINING POINTS IN THE NYSTROM MATRIX