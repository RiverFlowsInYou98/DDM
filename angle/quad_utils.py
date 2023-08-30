import numpy as np

# utility function, Gauss-Legendre quadrature in 2D and 3D
# test result: not as robust as scipy.integrate.nquad for weird functions, but thousands times faster


def Gauss_quad_2D(f, quad_range, order=5):
    (xl, xu), (yl, yu) = quad_range
    quad_points_ref, quad_weights = np.polynomial.legendre.leggauss(order)
    x_quad_points = quad_points_ref * (xu - xl) / 2 + (xu + xl) / 2
    y_quad_points = quad_points_ref * (yu - yl) / 2 + (yu + yl) / 2
    xv, yv = np.meshgrid(x_quad_points, y_quad_points, sparse=True, indexing="ij")
    weights = quad_weights[:, None] * quad_weights[None, :]
    return np.sum(np.multiply(f(xv, yv), weights)) * (xu - xl) * (yu - yl) / 4


def Gauss_quad_3D(f, quad_range, order=5):
    (xl, xu), (yl, yu), (zl, zu) = quad_range
    quad_points_ref, quad_weights = np.polynomial.legendre.leggauss(order)
    x_quad_points = quad_points_ref * (xu - xl) / 2 + (xu + xl) / 2
    y_quad_points = quad_points_ref * (yu - yl) / 2 + (yu + yl) / 2
    z_quad_points = quad_points_ref * (zu - zl) / 2 + (zu + zl) / 2
    xv, yv, zv = np.meshgrid(x_quad_points, y_quad_points, z_quad_points, sparse=True, indexing="ij")
    weights = quad_weights[:, None, None] * quad_weights[None, :, None] * quad_weights[None, None, :]
    return np.sum(np.multiply(f(xv, yv, zv), weights)) * (xu - xl) * (yu - yl) * (zu - zl) / 8


def Gauss_quad_nD(f, quad_range, orders=5):
    quad_range = np.array(quad_range)
    d = len(quad_range)
    if np.ndim(orders) == 0:
        orders = orders * np.ones((d,), dtype=int)
    else:
        assert len(orders) == d
    quad_points = []
    quad_weights = []
    for i in range(d):
        quad_points_ref, quad_weights_ref = np.polynomial.legendre.leggauss(orders[i])  
        quad_points.append(quad_points_ref * (quad_range[i, 1] - quad_range[i, 0]) / 2 + (quad_range[i, 1] + quad_range[i, 0]) / 2)
        quad_weights.append(quad_weights_ref)
    grids = np.meshgrid(*quad_points, sparse=True, indexing="ij")
    _weights = np.meshgrid(*quad_weights, indexing="ij")
    weights = np.prod(np.array(_weights), axis=0)
    return np.sum(np.multiply(f(*grids), weights)) * np.prod(quad_range[:, 1] - quad_range[:, 0]) / 2**d
