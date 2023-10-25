#ifndef QUADRATURE_H
#define QUADRATURE_H

#include <vector>


// Function prototype for Gauss-Legendre quadrature
double gauss_legendre_quadrature(
    double (*integrand)(double, double, double, double),
    const std::vector<std::vector<double>>& bounds);

#endif // QUADRATURE_H
