from math import exp, pi, sqrt
import time


def density_upper(t, mu, a, b, x0, trunc_num=100):
    tau = mu + b
    a1 = a - x0
    factor = t ** (-1.5) * exp(a1 * tau - 0.5 * tau**2 * t - b / (2 * a) * a1**2) / sqrt(2 * pi)
    result = 0
    for j in range(trunc_num):
        rj = (2 * j + 1) * a - (-1) ** j * x0
        term = (-1) ** j * rj * exp(0.5 * (b / a - 1 / t) * rj**2)
        if abs(term) < 1e-20:
            break
        result += term
    return result * factor


def density_lower(t, mu, a, b, x0, trunc_num=100):
    """
    First passage time density on the lower boundary
    """
    tau = -mu + b
    a1 = a + x0
    factor = t ** (-1.5) * exp(a1 * tau - 0.5 * tau**2 * t - b / (2 * a) * a1**2) / sqrt(2 * pi)
    result = 0
    for j in range(trunc_num):
        rj = (2 * j + 1) * a + (-1) ** j * x0
        term = (-1) ** j * rj * exp(0.5 * (b / a - 1 / t) * rj**2)
        if abs(term) < 1e-20:
            break
        result += term
    return result * factor


def density_vertical(x, mu, a, b, x0, T, trunc_num=100):
    """
    exit density on the vertical boundary
    """
    x = x - x0
    factor = exp(mu * x - 0.5 * mu**2 * T) / sqrt(T)
    result = exp(-0.5 * x * x / T) / sqrt(2 * pi)
    for j in range(1, trunc_num):
        t1 = 4 * b * j * (2 * a * j + x0) - (x - 4 * a * j) ** 2 / (2 * T)
        t2 = 4 * b * j * (2 * a * j - x0) - (x + 4 * a * j) ** 2 / (2 * T)
        t3 = 2 * b * (2 * j - 1) * (2 * a * j - a + x0) - (x + (4 * j - 2) * a + 2 * x0) ** 2 / (2 * T)
        t4 = 2 * b * (2 * j - 1) * (2 * a * j - a - x0) - (x - (4 * j - 2) * a + 2 * x0) ** 2 / (2 * T)
        term = exp(t1) + exp(t2) - exp(t3) - exp(t4)
        if abs(term) < 1e-20:
            break
        result += term / sqrt(2 * pi)
    return result * factor


if __name__ == "__main__":
    mu = 1.0
    a = 1.5
    b = 0.3
    x0 = -0.1

    t = 1.0
    x = 0
    
    start1 = time.time()
    for i in range(1000):
        result1 = density_upper(t, mu, a, b, x0)
    end1 = time.time()
    elapsed_time1 = (end1 - start1) * 10**9 / 1000

    start2 = time.time()
    for i in range(1000):
        result2 = density_lower(t, mu, a, b, x0)
    end2 = time.time()
    elapsed_time2 = (end2 - start2) * 10**9 / 1000
    start3 = time.time()
    for i in range(1000):
        result3 = density_vertical(x, mu, a, b, x0, t)
    end3 = time.time()
    elapsed_time3 = (end3 - start3) * 10**9 / 1000

    print("Result:", result1, result2, result3)
    print("Execution Time:", elapsed_time1, elapsed_time2, elapsed_time3, "nanoseconds")
