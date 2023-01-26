class calc_alpha_greedy_n:
    """Determines alpha co-efficient of each entry in `X` by counting the number of times each point
    (from the `n_posterior_samples` taken) appears within the `n_optimisation_points`.
    This differs from "Greedy Tau" sampling as here the top points are consideed regardless of their ability to breach
    a threshold value.

    Notes
    -----
    Implemented as callable object to allow for different optimisation objectives to be specified while maintaining
    API contract.

    References
    ----------
    See arXiv:2009.05418 "Bayesian Screening: Multi-test Bayesian Optimization Applied to in silico
    Material Screening" for further discussion of the method.

    Attributes
    ----------
    n_posterior_samples : int (default = 100)
        Number of times a sample is to be drawn from the posterior distribution.

    n_optimisation_points : int (default = 100)
        Number of points which the Greedy N algorithm is attempting to optimise for.

    """

    def __init__(self, n_posterior_samples=100, n_optimisation_points=100):
        """
        Parameters
        ----------
        n_posterior_samples : int (default = 100)
            Number of times a sample is to be drawn from the posterior distribution.

        n_optimisation_points : int (default = 100)
            Number of points which the Greedy N algorithm is attempting to optimise for.
        """
        self.n_posterior_samples = int(n_posterior_samples)
        self.n_optimisation_points = int(n_optimisation_points)

    @staticmethod
    def _calc_alpha(posterior, n_optimisation_points):
        """Determine alpha values by incrementing by one each time index is in top 100 posterior samples.

        Parameters
        ----------
        posterior  : np.ndarray
            shape(n_entries, n_posterior_samples)
            Multi dimensional array where each coloumn is a posterior sampling of each row.

        n_optimisation_points : int
            Number of points to optimise for (i.e. which threshold of posterior values to increment on).

        Returns
        -------
        alpha : np.ndarray
            shape(n_entries, )
            Alpha values for each entry in posterior.
        """
        a, b = posterior.shape
        alpha = np.zeros(a)
        for j in range(b):
            # argpartition saves sorting whole array as small `k` (~100) for large array size (typically >= 100,000)
            top_n_ind = np.argpartition(posterior[:, j], -n_optimisation_points)[-n_optimisation_points:]
            alpha[top_n_ind] += 1
        return alpha

    def __call__(self, model, X):
        # See object doccumentation string for doc string.
        posterior = access_posterior(model, X, self.n_posterior_samples)
        alpha = self._calc_alpha(posterior, self.n_optimisation_points)
        return alpha
