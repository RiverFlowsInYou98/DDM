import numpy as np
import itertools
from scipy.special import logsumexp
import scipy.sparse as sp
import networkx as nx
from utils import *


class approx_hmc(nx.DiGraph):
    """
    Markov chain approximation (homogeneous case)
    """

    def __init__(self, mu, sigma, a, z, dt, Nx, verbose=True) -> None:
        self.mu = mu  # drift coeff (constant)
        self.sigma = sigma  # diffusion coeff (constant)
        self.a = a  # upper boundary
        self.z = z * a  # starting point
        self.Nx = Nx  # num of space steps

        self.dt = dt
        dx = a / Nx
        self.dx = dx

        self.idx_z = int(round(z / dx))  # index of starting point
        self.init_dist = np.zeros(self.Nx + 2)
        self.init_dist[self.idx_z] = 1

        # transition probability matrix
        # 0, 1, ..., Nx are transient states
        # Nx+1 is the absorbing state
        m1 = mu * dt
        m2 = (mu * dt) ** 2 + sigma ** 2 * dt
        self.p1 = (m2 / dx ** 2 + m1 / dx) / 2
        self.p2 = (m2 / dx ** 2 - m1 / dx) / 2
        assert self.p1 + self.p2 < 1, "p+=%.5f, p0=%.5f, p-=%.5f" % (
            self.p1,
            1 - self.p1 - self.p2,
            self.p2,
        )
        self.Construct_AdjMat()

        if verbose:
            self.summary()

    def Construct_AdjMat(self):
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
        super(approx_hmc, self).__init__(self.AdjMat)

    def ExitDist(self, T):
        """
        compute the full distribution of X[T]
        where T is the first passage time
        by MATRIX MULTIPLICATION
        """
        dist_Xt = self.init_dist
        idx_T = int(round(T / self.dt))
        for t_step in range(idx_T):
            dist_Xt = dist_Xt @ self.AdjMat
        return dist_Xt

    def ExitProb_dp1(self, T, s):
        """
        compute the probability of P(X[T]=s)
        where t is the first passage time
        by DYNAMIC PROGRAMMING based on SPARSE ADJACENCY MATRIX
        s: value in [0, a]
        """
        idx_T = int(round(T / self.dt))
        idx_s = int(round(s / self.dx))
        table = np.zeros((self.Nx + 2, idx_T))
        table[:, [idx_T - 1]] = self.AdjMat[:, [idx_s]].toarray()
        for t_step in range(idx_T - 2, -1, -1):
            table[:, [t_step]] = self.AdjMat @ table[:, [t_step + 1]]
        return table[:, 0] @ self.init_dist

    def ExitProb_dp2(self, T, s):
        """
        compute the probability of P(X[T]=s)
        where t is the first passage time
        by DYNAMIC PROGRAMMING based on AJACENCY LIST
        s: value in [0, a]
        """
        idx_T = int(round(T / self.dt))
        idx_s = int(round(s / self.dx))
        table = np.zeros((self.Nx + 2, idx_T))
        table[:, [idx_T - 1]] = self.AdjMat[:, [idx_s]].toarray()
        for t_step in range(idx_T - 2, -1, -1):
            for u, v, w in self.in_edges(data=True):
                table[u, t_step] += w["weight"] * table[v, t_step + 1]
        return table[:, 0] @ self.init_dist

    def ExitProb_logdp1(self, T, s):
        """
        compute the probability of P(X[T]=s) in LOG scale
        where t is the first passage time
        by DYNAMIC PROGRAMMING based on ADJACENCY MATRIX
        s: value in [0, a]
        """
        idx_T = int(round(T / self.dt))
        idx_s = int(round(s / self.dx))
        logP = np.log(self.AdjMat.todense())
        logtable = np.zeros((self.Nx + 2, idx_T))
        logtable[:, [idx_T - 1]] = logP[:, [idx_s]]
        for t_step in range(idx_T - 2, -1, -1):
            logtable[:, [t_step]] = logdotexp(logP, logtable[:, [t_step + 1]])
        logprob = logdotexp(np.log(self.init_dist).reshape(1, -1), logtable[:, [0]])
        return np.squeeze(np.exp(logprob))[()]

    def ExitProb_logdp2(self, T, s):
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
        scaled_table[:, [idx_T - 1]] = self.AdjMat[:, [idx_s]].toarray() / np.exp(r)
        for t_step in range(idx_T - 2, -1, -1):
            b = np.sum(self.AdjMat @ scaled_table[:, [t_step + 1]])
            r = r + np.log(b)
            scaled_table[:, [t_step]] = self.AdjMat @ scaled_table[:, [t_step + 1]] / b
        return scaled_table[:, 0] * np.exp(r) @ self.init_dist

    def ExitProb_test(self, T, s):
        """
        compute the probability of P(X[T]=s)
        where t is the first passage time
        by BRUTAL FORCE (only used for testing!)
        s: value in [0, a]
        """
        idx_T = int(round(T / self.dt))
        idx_s = int(round(s / self.dx))
        ranges = [range(self.Nx + 2)] * (idx_T - 1)
        prob = 0
        for xs in itertools.product(*ranges):
            prob_traj = self.AdjMat[self.idx_z, xs[0]] * self.AdjMat[xs[-1], idx_s]
            for l in range(len(xs) - 1):
                prob_traj *= self.AdjMat[xs[l], xs[l + 1]]
            prob += prob_traj
        return prob

    def summary(self):
        print("mu: %.3f" % self.mu)
        print("sigma: %.3f" % self.sigma)
        print("a: %.3f" % self.a)
        print("z: %.3f" % self.z)
        print("dt: %.5f" % self.dt)
        print("dx: %.5f" % self.dx)
        print("Nx: %d" % self.Nx)
        print("shape of P: " + str(self.AdjMat.shape))
        print(
            "p+, p0, p-: %.5f, %.5f, %.5f" % (self.p1, 1 - self.p1 - self.p2, self.p2)
        )


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
