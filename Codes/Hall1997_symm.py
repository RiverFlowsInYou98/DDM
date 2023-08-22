import numpy as np


def normpdf(x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-(x**2) / 2)


def density_horiz_symm(t, mu, a, b, trunc_num=100, bdy="upper"):
    """
    First passage time density on the horizental boundaries
    TBD: logsumexp trick?
    """
    if bdy == "upper":
        tau = mu + b
    elif bdy == "lower":
        tau = -mu + b
    else:
        raise ValueError
    factor = (
        t ** (-1.5)
        * np.exp(a * tau - 0.5 * tau**2 * t - a * b / 2)
        / np.sqrt(2 * np.pi)
    )
    result = 0
    for j in range(trunc_num):
        term = (
            (-1) ** j
            * (2 * j + 1)
            * a
            * np.exp(0.5 * (a * b - a**2 / t) * (2 * j + 1) ** 2)
        )
        if np.max(np.abs(term)) < 1e-20:
            break
        result += term
        # print("{:>.8e}".format(term * factor), end="\t")
        # print("{:>.8e}".format(result * factor))
    return result * factor


def density_vertical_symm(x, mu, a, b, t0, trunc_num=100):
    factor = np.exp(mu * x - 0.5 * mu**2 * t0) / np.sqrt(t0)
    result = normpdf(x / np.sqrt(t0))
    for j in range(1, trunc_num):
        t1 = 4 * a * x / t0 * j - x**2 / (2 * t0)
        t2 = -4 * a * x / t0 * j - x**2 / (2 * t0)
        t3 = (
            (-8 * a * b - 4 * a / t0 * (x - 2 * a)) * j
            + 2 * a * b
            - (x - 2 * a) ** 2 / (2 * t0)
        )
        t4 = (
            (-8 * a * b + 4 * a / t0 * (x + 2 * a)) * j
            + 2 * a * b
            - (x + 2 * a) ** 2 / (2 * t0)
        )
        factor2 = np.exp((8 * a * b - 8 * a**2 / t0) * j**2) / np.sqrt(2 * np.pi)
        term =  (np.exp(t1) + np.exp(t2) - np.exp(t3) - np.exp(t4)) * factor2
        if np.max(np.abs(term)) < 1e-20:
            break
        # print("{:>.8e}".format(np.exp(t1) * factor2), end="\t")
        # print("{:>.8e}".format(np.exp(t2) * factor2), end="\t")
        # print("{:>.8e}".format(-np.exp(t3) * factor2), end="\t")
        # print("{:>.8e}".format(-np.exp(t4) * factor2), end="\t")
        # print("{:>.8e}".format(term))
        result += term
    return result * factor


# def density_vertical2(x, mu, a, theta, t0, trunc_num=100, debug=False):
#     c = 2 * a
#     b = theta
#     factor = np.exp(mu * x - 0.5 * mu**2 * t0) / np.sqrt(t0)
#     result = normpdf(x / np.sqrt(t0)) * factor
#     for j in range(1, trunc_num):
#         term1 = np.exp(4 * b * j**2 * c) * normpdf((x - 2 * j * c) / np.sqrt(t0))
#         term2 = np.exp(4 * b * j**2 * c) * normpdf((x + 2 * j * c) / np.sqrt(t0))
#         term3 = np.exp(2 * b * (2 * j - 1) * (j * c - a)) * normpdf(
#             (x + 2 * j * c - 2 * a) / np.sqrt(t0)
#         )
#         term4 = np.exp(2 * b * (2 * j - 1) * (j * c - a)) * normpdf(
#             (x - 2 * j * c + 2 * a) / np.sqrt(t0)
#         )
#         term = term1 + term2 - term3 - term4
#         result += term * factor
#         if debug:
#             print(term * factor)
#     return result
