import numpy as np


def normpdf(x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-(x**2) / 2)


def density_horiz(t, mu, a, b, trunc_num=100, bdy="upper"):
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
        result += term
        # print("{:>.8e}".format(term* factor), end="\t")
        # print("{:>.8e}".format(result* factor))
    return result * factor


def density_horiz_debug(t, mu, a, theta, trunc_num=100, bdy="upper", debug=False):
    c = 2 * a
    b = theta
    if bdy == "upper":
        tau = mu + theta
    elif bdy == "lower":
        tau = -mu + theta
    else:
        raise ValueError
    result = 0
    # if debug:
    #     print("even_term\t   odd_term\t\tterm\t\tresult")
    for k in range(trunc_num):
        r_even = 2 * k * c + a
        r_odd = (2 * k + 1) * c + a
        even_term = (
            np.exp(b / c * (r_even**2 - a**2))
            * r_even
            * normpdf(r_even / np.sqrt(t))
        )
        odd_term = (
            -np.exp(b / c * (r_odd**2 - a**2)) * r_odd * normpdf(r_odd / np.sqrt(t))
        )
        term = even_term + odd_term
        # if debug:
        #     print(
        #         "{:>.8e}".format(
        #             t ** (-1.5) * np.exp(a * tau - 0.5 * tau**2 * t) * even_term
        #         ),
        #         end="\t",
        #     )
        #     print(
        #         "{:>.8e}".format(
        #             t ** (-1.5) * np.exp(a * tau - 0.5 * tau**2 * t) * odd_term
        #         ),
        #         end="\t",
        #     )
        #     print(
        #         "{:>.8e}".format(
        #             t ** (-1.5) * np.exp(a * tau - 0.5 * tau**2 * t) * term
        #         ),
        #         end="\t",
        #     )
        result += t ** (-1.5) * np.exp(a * tau - 0.5 * tau**2 * t) * term
        # if debug:
        #     print("{:>.8e}".format(result))
    return result


def density_vertical(x, mu, a, theta, t0, trunc_num=100, debug=False):
    c = 2 * a
    b = theta
    factor = np.exp(mu * x - 0.5 * mu**2 * t0) / np.sqrt(t0)
    result = normpdf(x / np.sqrt(t0)) * factor
    for j in range(trunc_num):
        term1 = np.exp(4 * b * j**2 * c) * normpdf((x - 2 * j * c) / np.sqrt(t0))
        term2 = np.exp(4 * b * j**2 * c) * normpdf((x + 2 * j * c) / np.sqrt(t0))
        term3 = np.exp(2 * b * (2 * j - 1) * (j * c - a)) * normpdf(
            (x + 2 * j * c - 2 * a) / np.sqrt(t0)
        )
        term4 = np.exp(2 * b * (2 * j - 1) * (j * c - a)) * normpdf(
            (x - 2 * j * c + 2 * a) / np.sqrt(t0)
        )
        term = term1 + term2 - term3 - term4
        result += term * factor
        if debug:
            print(term * factor)
    return result


def density_horiz2(t, mu, a, theta, trunc_num=100, bdy="upper", debug=False):
    c = 2 * a
    b = theta
    if bdy == "upper":
        tau = mu + theta
    elif bdy == "lower":
        tau = -mu + theta
    else:
        raise ValueError
    result = 0
    factor = (
        t ** (-1.5)
        * np.exp(a * tau - 0.5 * tau**2 * t - b / c * a**2)
        / np.sqrt(2 * np.pi)
    )
    # if debug:
    #     print("even_term\t   odd_term\t\tterm\t\tresult")
    for k in range(trunc_num):
        t1 = np.exp((b / c - 1 / (2 * t)) * (2 * k * c + a) ** 2)
        t2 = (
            2 * k * c
            + a
            - ((2 * k + 1) * c + a)
            * np.exp((b - c / (2 * t)) * ((4 * k + 1) * c + 2 * a))
        )

        # r_even = 2 * k * c + a
        # r_odd = (2 * k + 1) * c + a
        # even_term = r_even * np.exp(
        #     (b / c - 1 / (2 * t)) * r_even**2 - b / c * a**2
        # )
        # odd_term = -r_odd * np.exp((b / c - 1 / (2 * t)) * r_odd**2 - b / c * a**2)
        # term = even_term + odd_term
        # if debug:
        #     print(
        #         "{:>.8e}".format(
        #             t ** (-1.5) * np.exp(a * tau - 0.5 * tau**2 * t) * even_term
        #         ),
        #         end="\t",
        #     )
        #     print(
        #         "{:>.8e}".format(
        #             t ** (-1.5) * np.exp(a * tau - 0.5 * tau**2 * t) * odd_term
        #         ),
        #         end="\t",
        #     )
        #     print(
        #         "{:>.8e}".format(
        #             t ** (-1.5) * np.exp(a * tau - 0.5 * tau**2 * t) * term
        #         ),
        #         end="\t",
        #     )
        result += t1 * t2
        # if debug:
        #     print("{:>.8e}".format(result))
    return factor * result


def density_vertical(x, mu, a, theta, t0, trunc_num=100, debug=False):
    c = 2 * a
    b = theta
    factor = np.exp(mu * x - 0.5 * mu**2 * t0) / np.sqrt(t0)
    result = normpdf(x / np.sqrt(t0)) * factor
    for j in range(trunc_num):
        term1 = np.exp(4 * b * j**2 * c) * normpdf((x - 2 * j * c) / np.sqrt(t0))
        term2 = np.exp(4 * b * j**2 * c) * normpdf((x + 2 * j * c) / np.sqrt(t0))
        term3 = np.exp(2 * b * (2 * j - 1) * (j * c - a)) * normpdf(
            (x + 2 * j * c - 2 * a) / np.sqrt(t0)
        )
        term4 = np.exp(2 * b * (2 * j - 1) * (j * c - a)) * normpdf(
            (x - 2 * j * c + 2 * a) / np.sqrt(t0)
        )
        term = term1 + term2 - term3 - term4
        result += term * factor
        if debug:
            print(term * factor)
    return result
