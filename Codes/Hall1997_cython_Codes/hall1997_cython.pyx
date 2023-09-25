cimport cython
from libc.math cimport pi, sqrt, exp, pow, fabs


@cython.cdivision(True)
def normpdf(float x):
    return 1 / sqrt(2 * pi) * exp(-pow(x, 2) / 2)

@cython.cdivision(True)
def density_upper_cython(float t, float mu, float a, float b, float x0, unsigned int trunc_num=100):
    """
    First passage time density on the upper boundary
    """
    cdef float tau, a1, factor, result, rj, term
    cdef unsigned int j
    tau = mu + b
    a1 = a - x0
    factor = pow(t, -1.5) * exp(a1 * tau - 0.5 * pow(tau, 2) * t - b / (2 * a) * pow(a1, 2)) / sqrt(2 * pi)
    result = 0
    for j in range(trunc_num):
        rj = (2 * j + 1) * a - pow(-1, j) * x0
        term = pow(-1, j) * rj * exp(0.5 * (b / a - 1 / t) * pow(rj, 2))
        if fabs(term) < 1e-20:
            break
        result += term
    return result * factor

@cython.cdivision(True)
def density_lower_cython(float t, float mu, float a, float b, float x0, unsigned int trunc_num=100):
    """
    First passage time density on the upper boundary
    """
    cdef float tau, a1, factor, result, rj, term
    cdef unsigned int j
    tau = -mu + b
    a1 = a + x0
    factor = pow(t, -1.5) * exp(a1 * tau - 0.5 * pow(tau, 2) * t - b / (2 * a) * pow(a1, 2)) / sqrt(2 * pi)
    result = 0
    for j in range(trunc_num):
        rj = (2 * j + 1) * a + pow(-1, j) * x0
        term = pow(-1, j) * rj * exp(0.5 * (b / a - 1 / t) * pow(rj, 2))
        if fabs(term) < 1e-20:
            break
        result += term
    return result * factor


@cython.cdivision(True)
def density_vertical_cython(float x, float mu, float a, float b, float x0, float T, unsigned int trunc_num=100):
    """
    exit density on the vertical boundary
    """
    cdef float t1, t2, t3, t4, term, factor
    cdef unsigned int j
    x = x - x0
    factor = exp(mu * x - 0.5 * pow(mu, 2) * T) / sqrt(T)
    result = 1 / sqrt(2 * pi) * exp(-pow(x, 2) / (2*T))
    for j in range(1, trunc_num):
        t1 = 4 * b * j * (2 * a * j + x0) - pow(x - 4 * a * j, 2) / (2 * T)
        t2 = 4 * b * j * (2 * a * j - x0) - pow(x + 4 * a * j, 2) / (2 * T)
        t3 = 2 * b * (2 * j - 1) * (2 * a * j - a + x0) - pow(x + (4 * j - 2) * a + 2 * x0, 2) / (2 * T)
        t4 = 2 * b * (2 * j - 1) * (2 * a * j - a - x0) - pow(x - (4 * j - 2) * a + 2 * x0, 2) / (2 * T)
        term = exp(t1) + exp(t2) - exp(t3) - exp(t4)
        if fabs(term) < 1e-20:
            break
        result += term / sqrt(2 * pi)
    return result * factor
