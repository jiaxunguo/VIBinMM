import numpy as np

from sklearn.mixture import BayesianGaussianMixture
from numpy.linalg import inv
from numpy import linalg as LA


class VIBinMM:

    def __init__(self, n_components=1, max_iter=100, thred=1e-2, tol=1e-3, n_init=1, init_params='random', \
                 mean_precision_prior=100, mean_prior=None, random_state=None):
        '''
        mean_prior: array-like, shape(n_features,), default=[0,...,0]
        '''

        self.D = None
        self.J = n_components

        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.init_params = init_params
        self.mean_precision_prior = mean_precision_prior
        self.mean_prior = mean_prior
        self.random_state = random_state

        self.thred = thred

        self.weight_bhm = None
        self.M_bhm = None
        self.Z_bhm = None

        self.Mus_gau = None
        self.Covs_gau = None

        self.mu = None

        self.gau_model = None

    def init_top_params(self, data):
        self.D = data.shape[1]

        self.weight_bhm = np.ones(self.J) / self.J
        self.M_bhm = np.ones((self.J, self.D, self.D))
        self.Z_bhm = np.ones((self.J, self.D))

        self.mu = np.ones((self.J, self.D))
        self.mu = self.mu / np.linalg.norm(self.mu, axis=1)[:, np.newaxis]

        if self.mean_prior is None:
            self.mean_prior = np.zeros(self.D)

    def GMM2BinMM(self, data):

        bgm = BayesianGaussianMixture(n_components=self.J, tol=self.tol, init_params=self.init_params,
                                      mean_precision_prior=self.mean_precision_prior, \
                                      mean_prior=self.mean_prior, max_iter=self.max_iter, \
                                      n_init=self.n_init).fit(data)

        self.gau_model = bgm

        self.weight_bhm = bgm.weights_

        self.Mus_gau = bgm.means_
        self.Covs_gau = bgm.covariances_

        for i in range(self.J):
            _A = -0.5 * inv(bgm.covariances_[i])
            _Z, _invM = LA.eig(_A)

            self.Z_bhm[i] = _Z
            self.M_bhm[i] = inv(_invM)

            self.mu[i] = self.M_bhm[i, np.argmax(self.Z_bhm[i])]

        # Z tuning
        self.Z_bhm[self.Z_bhm == np.max(self.Z_bhm, axis=1)[:, np.newaxis]] = 0

        # M and Z sorting
        self.M_bhm = np.take_along_axis(self.M_bhm, np.argsort(np.repeat(np.expand_dims(self.Z_bhm, -1), self.Z_bhm.shape[1], axis=-1), axis=1), axis=1)
        self.Z_bhm = np.take_along_axis(self.Z_bhm, np.argsort(self.Z_bhm), axis=1)

    def update_params_byThred(self):

        mask = np.where(self.weight_bhm > self.thred)

        self.weight_bhm = self.weight_bhm[mask]
        self.M_bhm = self.M_bhm[mask]
        self.Z_bhm = self.Z_bhm[mask]

        self.Mus_gau = self.Mus_gau[mask]
        self.Covs_gau = self.Covs_gau[mask]

        self.mu = self.mu[mask]

        self.J = np.size(self.weight_bhm)

    def update_params_byWgtDesc(self):

        sorted_indices = np.argsort(self.weight_bhm)[::-1]

        self.weight_bhm = self.weight_bhm[sorted_indices]
        self.M_bhm = self.M_bhm[sorted_indices]
        self.Z_bhm = self.Z_bhm[sorted_indices]

        self.Mus_gau = self.Mus_gau[sorted_indices]
        self.Covs_gau = self.Covs_gau[sorted_indices]

        self.mu = self.mu[sorted_indices]

    def display_params(self):

        print('Weights: ', self.weight_bhm)
        print('Ms: ', self.M_bhm)
        print('Zs: ', self.Z_bhm)
        print('Mus: ', self.mu)

        print('Mus_gau: ', self.Mus_gau)
        print('Covs_gau: ', self.Covs_gau)

    def fit(self, data, verbose=0, wgtdesc=True):

        self.init_top_params(data)
        self.GMM2BinMM(data)
        self.update_params_byThred()

        if wgtdesc:
            self.update_params_byWgtDesc()

        if verbose:
            self.display_params()

        return self

    @staticmethod
    def _rv(E, K, n):
        '''
        https://github.com/edur409/Bingham_distributions
        The Bingham distribution from the principal directions and concentrations
        k1 and k2.  k1 and k2 must be negative numbers
        ################################
        #### Simulating from a Bingham distribution based on the principal directions
        #### inferred from a set of poles and their concentrations.
        ################################

        ######### Simulation using any symmetric matrix A
        xc,yc,zc=pbingham(n,E,k1,k2)
        ######### Output
        xc,yc,zc are the coordinates of unit vectors on the sphere
        '''
        from numpy.random import random as runif
        p = len(E) - 1

        lam = -K

        nsamp = 0
        X = []
        qa = len(lam)
        mu = np.zeros(qa)  # Zero means
        sigacginv = 1 + 2 * lam
        SigACG = np.sqrt(1 / (1 + 2 * lam))  # standard deviations
        Ntry = 0

        while nsamp < n:
            xsamp = False
            while (not xsamp):
                yp = np.random.normal(mu, SigACG, qa)
                y = yp / np.sqrt(np.sum(yp ** 2))
                lratio = -np.sum(y ** 2 * lam) - qa / 2 * np.log(qa) + 0.5 * (qa - 1) + qa / 2 * np.log(
                    np.sum(y ** 2 * sigacginv))
                if (np.log(runif(1)) < lratio):
                    X = np.append(X, y)
                    xsamp = True
                    nsamp = nsamp + 1
                Ntry = Ntry + 1
        x = X.reshape((n, qa))  # n normal vectors
        X = E.T @ x.T
        return X.T

    def rvs(self, size):

        size_by_classes = np.round(size * self.weight_bhm)
        size_by_classes[np.argmax(size_by_classes)] += size - np.sum(size_by_classes)
        size_by_classes = size_by_classes.astype(int)

        print('size_by_classes:', size_by_classes)

        assert np.sum(size_by_classes) == size

        labels = np.repeat(np.arange(len(size_by_classes)), size_by_classes)

        samples = []
        for i in range(self.J):
            samples.append(self._rv(self.M_bhm[i], self.Z_bhm[i], size_by_classes[i]))

        samples = np.vstack(samples)

        return samples, labels









