import numpy as np
import itertools
from scipy.special import logsumexp
import scipy.sparse as sp
from utils import *


def mu_pc(t, u1, u2, endpoints):
    jumps = endpoints[1:-1]
    return (-1) ** np.searchsorted(jumps, t) * (u1 - u2) / 2 + (u2 + u1) / 2


def mut_pl(t, u1, u2, endpoints):
    jumps = endpoints[1:-1]
    isi = np.diff(endpoints)
    idx = np.searchsorted(jumps, t)
    isi_t = np.hstack((isi[:idx], t - jumps[idx - 1])) if idx > 0 else t
    return np.sum(isi_t * np.resize([u1, u2], idx + 1))


class ihmc(object):
    """
    Markov chain approximation (inhomogeneous case)
    the drift is a piecewise constant alternating between 'u1' and u2'
    the time lags are exponential distributed
    """

    def __init__(self, mu1, mu2, sigma, a, z, dt, Nx, verbose=True) -> None:
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma = sigma  # diffusion coeff (constant)
        self.a = a  # upper boundary
        self.z = z  # starting point
        self.Nx = Nx  # num of space steps

        self.dt = dt
        dx = a / Nx
        self.dx = dx

        self.idx_z = int(round(z / dx))  # index of starting point
        self.init_dist = np.zeros(self.Nx + 2)
        self.init_dist[self.idx_z] = 1

    def Update_AdjMat(self, t, endpoints):
        """
        inhomogeneous transition probability matrix
        0, 1, ..., Nx are transient states
        Nx+1 is the absorbing state
        arguments:
        t - time
        endpoints: the jumping times of the current sample path
        """
        mu_t = lambda t: mu_pc(t, self.mu1, self.mu2, endpoints)
        mu = mu_t(t)
        m1 = mu * self.dt
        m2 = (mu * self.dt) ** 2 + self.sigma ** 2 * self.dt
        self.p1 = (m2 / self.dx ** 2 + m1 / self.dx) / 2
        self.p2 = (m2 / self.dx ** 2 - m1 / self.dx) / 2
        assert self.p1 + self.p2 < 1, "p+=%.5f, p0=%.5f, p-=%.5f" % (
            self.p1,
            1 - self.p1 - self.p2,
            self.p2,
        )
        AdjMat = sp.dok_matrix((self.Nx + 2, self.Nx + 2))
        nz_dict = {
            (0, self.Nx + 1): 1,
            (self.Nx, self.Nx + 1): 1,
            (self.Nx + 1, self.Nx + 1): 1,
        }
        for i in range(1, self.Nx):
            nz_dict[(i, i - 1)] = self.p2
            nz_dict[(i, i)] = 1 - self.p1 - self.p2
            nz_dict[(i, i + 1)] = self.p1
        dict.update(AdjMat, nz_dict)
        self.AdjMat = sp.csr_matrix(AdjMat)

    def ExitDist(self, T, endpoints):
        """
        compute the full distribution of X[T]
        where T is the first passage time
        by MATRIX MULTIPLICATION
        """
        dist_Xt = self.init_dist
        idx_T = int(round(T / self.dt))
        for t_step in range(idx_T):
            self.Update_AdjMat(t_step * self.dt, endpoints)
            dist_Xt = dist_Xt @ self.AdjMat
        return dist_Xt

    def ExitProb_dp(self, T, s, endpoints):
        """
        compute the probability of P(X[T]=s)
        where t is the first passage time
        by DYNAMIC PROGRAMMING based on SPARSE ADJACENCY MATRIX
        s: value in [0, a]
        """
        idx_T = int(round(T / self.dt))
        idx_s = int(round(s / self.dx))
        table = np.zeros((self.Nx + 2, idx_T))
        self.Update_AdjMat((idx_T - 1) * self.dt, endpoints)
        table[:, [idx_T - 1]] = self.AdjMat[:, [idx_s]].toarray()
        for t_step in range(idx_T - 2, -1, -1):
            self.Update_AdjMat(t_step * self.dt, endpoints)
            table[:, [t_step]] = self.AdjMat @ table[:, [t_step + 1]]
        return table[:, 0] @ self.init_dist

    def ExitProb_logdp(self, T, s, endpoints):
        """
        compute the probability of P(X[T]=s) with a EXP scaling
        where t is the first passage time
        by DYNAMIC PROGRAMMING based on ADJACENCY MATRIX
        s: value in [0, a]
        """
        idx_T = int(round(T / self.dt))
        idx_s = int(round(s / self.dx))
        scaled_table = np.zeros((self.Nx + 2, idx_T))
        r = 0
        self.Update_AdjMat((idx_T - 1) * self.dt, endpoints)
        scaled_table[:, [idx_T - 1]] = self.AdjMat[:, [idx_s]].toarray() / np.exp(r)
        for t_step in range(idx_T - 2, -1, -1):
            self.Update_AdjMat(t_step * self.dt, endpoints)
            b = np.sum(self.AdjMat @ scaled_table[:, [t_step + 1]])
            r = r + np.log(b)
            scaled_table[:, [t_step]] = self.AdjMat @ scaled_table[:, [t_step + 1]] / b
        return scaled_table[:, 0] * np.exp(r) @ self.init_dist
