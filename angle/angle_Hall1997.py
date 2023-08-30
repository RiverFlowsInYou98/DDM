import numpy as np
import sys
import gc

sys.path.append("../Codes/")
from Hall1997 import *
from quad_utils import *


def get_fptd_aAngle_gauss(t, mu_list, t_list, a, theta, x0, bdy):
    """
    Get the first passage time density on the upper/lower boundary
    for drift-diffusion model with piecewise constant drift rate and linear collapsing boundaries
    use multi-dimensional Gauss-Legendre quadrature rule, more efficient at each stage, but
    time complexity will grow exponentially with `d`, i.e. O(N1*N2*N3...)~O(N^d)
    Inputs:
    -- t: the time at which we want to evaluate the density
          t should be strictly greater than t_list[-1]
          Note: might have numerical issues if t is too close to t_list[-1]
    -- mu_list: a list of drift rates, one for each piecewise linear part. has length `d`
    -- t_list: a increasing list of times when the drift rates change, doesn't include 0. has length `d-1`
    -- a, theta: the upper boundary and lower boundary are given by `a - theta * t` and `-a + theta * t` respectively
    -- x0: starting point of the process
    -- bdy: takes value in {-1, +1}, representing the lower and upper boundary
    """
    # check the parameters
    # number of stages
    d = len(mu_list)
    assert len(t_list) == d - 1
    # two types of transition densities
    p = lambda x, t, y, s, mu: density_vertical(x, mu=mu, a=a - theta * s, b=theta, x0=y, T=t - s, trunc_num=100)
    if bdy == 1:
        f = lambda t, y, s, mu: density_upper(t - s, mu=mu, a=a - theta * s, b=theta, x0=y, trunc_num=100)
    elif bdy == -1:
        f = lambda t, y, s, mu: density_lower(t - s, mu=mu, a=a - theta * s, b=theta, x0=y, trunc_num=100)
    else:
        raise ValueError
    # if only one stage, i.e. drift rate is constant
    if d == 1:
        return f(t, x0, 0, mu_list[0])
    # if more than one stage, i.e. drift rate is piecewise constant
    else:
        assert all(i < j for i, j in zip(t_list, t_list[1:]))
        assert t_list[0] > 0
        assert t > t_list[-1]
        upper_bdy = lambda t: a - theta * t
        lower_bdy = lambda t: -a + theta * t

        def integrand(*xs):
            """
            the integrand to be integrated numerically to get the result
            `xs` is [x1, x2, ..., x{d-1}], has length d-1
            """
            assert len(xs) == d - 1
            result = p(xs[0], t_list[0], x0, 0, mu_list[0]) * f(t, xs[d - 2], t_list[d - 2], mu_list[d - 1])
            for i in range(1, d - 1):
                result = np.multiply(result, p(xs[i], t_list[i], xs[i - 1], t_list[i - 1], mu_list[i]))
            return result

        int_range = []
        for i in range(d - 1):
            int_range.append([lower_bdy(t_list[i]), upper_bdy(t_list[i])])
        int_orders = [5] * (d - 2)
        int_orders.append(20)
        return Gauss_quad_nD(integrand, int_range, int_orders)


def get_fptd_aAngle_seq(t, mu_list, t_list, a, theta, x0, bdy):
    """
    Get the first passage time density on the upper/lower boundary
    for drift-diffusion model with piecewise constant drift rate and linear collapsing boundaries
    use trapezoidal rule to proceed sequentially, not as efficient as Gauss quadrature for one stage
    but some computations are shared so the time complexity grows linearly with number of stages
    i.e. O(N1*N2+N2*N3+...)~O(d*N^2)
    Inputs:
    -- t: the time at which we want to evaluate the density
          t should be strictly greater than t_list[-1]
          Note: might have numerical issues if t is too close to t_list[-1]
    -- mu_list: a list of drift rates, one for each piecewise linear part. has length `d`
    -- t_list: a increasing list of times when the drift rates change, doesn't include 0. has length `d-1`
    -- a, theta: the upper boundary and lower boundary are given by `a - theta * t` and `-a + theta * t` respectively
    -- x0: starting point of the process
    -- bdy: takes value in {-1, +1}, representing the lower and upper boundary
    """
    # check the parameters
    # number of stages
    d = len(mu_list)
    assert len(t_list) == d - 1
    # two types of transition densities
    p = lambda x, t, y, s, mu: density_vertical(x, mu=mu, a=a - theta * s, b=theta, x0=y, T=t - s, trunc_num=100)
    if bdy == 1:
        f = lambda t, y, s, mu: density_upper(t - s, mu=mu, a=a - theta * s, b=theta, x0=y, trunc_num=100)
    elif bdy == -1:
        f = lambda t, y, s, mu: density_lower(t - s, mu=mu, a=a - theta * s, b=theta, x0=y, trunc_num=100)
    else:
        raise ValueError
    # if only one stage, i.e. drift rate is constant
    if d == 1:
        return f(t, x0, 0, mu_list[0])
    # if more than one stage, i.e. drift rate is piecewise constant
    else:
        assert all(i < j for i, j in zip(t_list, t_list[1:]))
        assert t_list[0] > 0
        assert t > t_list[-1]
        upper_bdy = lambda t: a - theta * t
        lower_bdy = lambda t: -a + theta * t
        # proceed iteratively to get the spatial distribution at t=t_list[-1]=t_list[d-2]
        # integrated over x1, x2, ..., x{d-2}
        # careful about the index: x1, x2, ..., x{d-2}, x{d-1} corresponds to t=t_list[0], t_list[1], ..., t_list[d-3], t_list[d-2]
        xs = np.linspace(lower_bdy(t_list[0]), upper_bdy(t_list[0]), 100)
        pv_seq_list = []
        for k, x in enumerate(xs):
            pv_seq_list.append(p(x, t_list[0], x0, 0, mu_list[0]))
        xs_old = xs.copy()
        pv_seq_list_old = pv_seq_list.copy()
        for n in range(1, d - 1):
            xs = np.linspace(lower_bdy(t_list[n]), upper_bdy(t_list[n]), 100)
            for k, x in enumerate(xs):
                result_v = 0
                for i in range(1, len(xs_old) - 1):
                    result_v += pv_seq_list_old[i] * p(x, t_list[n], xs_old[i], t_list[n - 1], mu_list[n])
                result_v *= xs_old[1] - xs_old[0]
                pv_seq_list[k] = result_v
            xs_old = xs.copy()
            pv_seq_list_old = pv_seq_list.copy()
        # proceed the final step, integrated over x{d-1}
        result = 0
        for i in range(1, len(xs_old) - 1):
            result += pv_seq_list[i] * f(t, xs_old[i], t_list[d - 2], mu_list[d - 1])
        result *= xs_old[1] - xs_old[0]
        return result

