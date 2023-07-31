import numpy as np


def normpdf(x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-(x**2) / 2)


def density(mu, a, theta, t, trunc_num=100, bdy="upper", debug=False):
    c = 2 * a
    b = theta
    if bdy == "upper":
        tau = mu + theta
    elif bdy == "lower":
        tau = -mu + theta
    else:
        raise ValueError
    result = 0
    for k in range(trunc_num):
        r1 = 2 * k * c + a
        r2 = (2 * k + 1) * c + a
        term = np.exp(b / c * (r1**2 - a**2)) * r1 * normpdf(
            r1 / np.sqrt(t)
        ) - np.exp(b / c * (r2**2 - a**2)) * r2 * normpdf(r2 / np.sqrt(t))
        if debug:
            print(t ** (-1.5) * np.exp(a * tau - 0.5 * tau**2 * t) * term)
        result += t ** (-1.5) * np.exp(a * tau - 0.5 * tau**2 * t) * term
    return result
