from numpy import exp, sqrt, pi
import numpy as np
from scipy.special import logsumexp


def normpdf(x):
    return 1 / sqrt(2 * pi) * exp(-(x**2) / 2)


def density_upper(t, mu, a, b, x0, trunc_num=100):
    """
    First passage time density on the upper boundary
    """
    tau = mu + b
    a1 = a - x0
    factor = (
        t ** (-1.5)
        * exp(a1 * tau - 0.5 * tau**2 * t - b / (2 * a) * a1**2)
        / sqrt(2 * pi)
    )
    result = 0
    for j in range(trunc_num):
        rj = (2 * j + 1) * a - (-1) ** j * x0
        term = (-1) ** j * rj * exp(0.5 * (b / a - 1 / t) * rj**2)
        if np.max(np.abs(term)) < 1e-20:
            break
        result += term
    return result * factor


def density_lower(t, mu, a, b, x0, trunc_num=100):
    """
    First passage time density on the lower boundary
    """
    tau = -mu + b
    a1 = a + x0
    factor = (
        t ** (-1.5)
        * exp(a1 * tau - 0.5 * tau**2 * t - b / (2 * a) * a1**2)
        / sqrt(2 * pi)
    )
    result = 0
    for j in range(trunc_num):
        rj = (2 * j + 1) * a + (-1) ** j * x0
        term = (-1) ** j * rj * exp(0.5 * (b / a - 1 / t) * rj**2)
        if np.max(np.abs(term)) < 1e-20:
            break
        result += term
    return result * factor


def density_vertical(x, mu, a, b, x0, T, trunc_num=100, if_logsumexp=True):
    """
    doesn't work, haven't figured out why
    """
    x = x - x0
    factor = exp(mu * x - 0.5 * mu**2 * T) / sqrt(T)
    result = normpdf(x / sqrt(T))
    for j in range(1, trunc_num):
        t1 = 4 * b * j * (2 * a * j + x0) - (x - 4 * a * j) ** 2 / (2 * T)
        t2 = 4 * b * j * (2 * a * j - x0) - (x + 4 * a * j) ** 2 / (2 * T)
        t3 = 2 * b * (2 * j - 1) * (2 * a * j - a + x0) - (x + (4 * j - 2) * a + 2 * x0) ** 2 / (2 * T)
        t4 = 2 * b * (2 * j - 1) * (2 * a * j - a - x0) - (x - (4 * j - 2) * a + 2 * x0) ** 2 / (2 * T)
        if if_logsumexp:
            logterm, sign = logsumexp([t1, t2, t3, t4], b=[1, 1, -1, -1], return_sign=True)
            if logterm < -20:
                break
            result += sign * exp(logterm) / sqrt(2 * pi)
        else:
            term = exp(t1) + exp(t2) - exp(t3) - exp(t4)
            if np.max(np.abs(term)) < 1e-50:
                break
            result += term / sqrt(2 * pi)
    return result * factor
