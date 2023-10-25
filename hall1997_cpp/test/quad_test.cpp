#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>

// Define the number of quadrature points (5th order for x, y, z; 20th order for w)
const int num_points_xyz = 5;
const int num_points_w = 20;

// Define the quadrature points and weights (manually)
const double quadrature_points_xyz[num_points_xyz] = {
    -0.9061798459,
    -0.5384693101,
    0.0,
    0.5384693101,
    0.9061798459};

const double quadrature_weights_xyz[num_points_xyz] = {
    0.2369268851,
    0.4786286705,
    0.5688888889,
    0.4786286705,
    0.2369268851};

const double quadrature_points_w[num_points_w] = {
    -0.9931285992,
    -0.9639719273,
    -0.9122344283,
    -0.8391169718,
    -0.7463319065,
    -0.6360536808,
    -0.5108670019,
    -0.3737060887,
    -0.2277858511,
    -0.0765265211,
    0.0765265211,
    0.2277858511,
    0.3737060887,
    0.5108670019,
    0.6360536808,
    0.7463319065,
    0.8391169718,
    0.9122344283,
    0.9639719273,
    0.9931285992};

const double quadrature_weights_w[num_points_w] = {
    0.0176140071,
    0.0406014298,
    0.0626720483,
    0.0832767410,
    0.1019301205,
    0.1181945319,
    0.1316886384,
    0.1420961093,
    0.1491729865,
    0.1527533871,
    0.1527533871,
    0.1491729865,
    0.1420961093,
    0.1316886384,
    0.1181945319,
    0.1019301205,
    0.0832767410,
    0.0626720483,
    0.0406014298,
    0.0176140071};

// Function to integrate (modify this function as needed)
double function_to_integrate(double x, double y, double z, double w)
{
    // Replace this with the function you want to integrate
    return x * x + y * y + y * z + exp(-z) * cos(w); // Example: integrating x^2 + y^2 + y*z + exp(-z) * cos(w)
}

// Function to perform Gauss-Legendre quadrature for a given integrand and bounds
double gauss_legendre_quadrature(
    double (*integrand)(double, double, double, double),
    const std::vector<std::vector<double>> &bounds)
{

    double integral = 0.0;

    for (int i = 0; i < num_points_xyz; i++)
    {
        for (int j = 0; j < num_points_xyz; j++)
        {
            for (int k = 0; k < num_points_xyz; k++)
            {
                for (int l = 0; l < num_points_w; l++)
                {
                    double xi = 0.5 * (bounds[0][1] - bounds[0][0]) * quadrature_points_xyz[i] + 0.5 * (bounds[0][1] + bounds[0][0]);
                    double yi = 0.5 * (bounds[1][1] - bounds[1][0]) * quadrature_points_xyz[j] + 0.5 * (bounds[1][1] + bounds[1][0]);
                    double zi = 0.5 * (bounds[2][1] - bounds[2][0]) * quadrature_points_xyz[k] + 0.5 * (bounds[2][1] + bounds[2][0]);
                    double wi = 0.5 * (bounds[3][1] - bounds[3][0]) * quadrature_points_w[l] + 0.5 * (bounds[3][1] + bounds[3][0]);
                    integral += quadrature_weights_xyz[i] * quadrature_weights_xyz[j] * quadrature_weights_xyz[k] * quadrature_weights_w[l] * integrand(xi, yi, zi, wi);
                }
            }
        }
    }
    double volume = 1.0;
    for (int i = 0; i < 4; i++)
    {
        volume *= 0.5 * (bounds[i][1] - bounds[i][0]);
    }

    integral *= volume;

    return integral;
}

int main()
{
    // Define the integration bounds as a 2D vector
    std::vector<std::vector<double>> bounds = {
        {-1, 1},     // [a1, b1]
        {-3, 2},     // [a2, b2]
        {-1.5, 2.5}, // [a3, b3]
        {1, 2}       // [a4, b4]
    };
    double result;
    // Start measuring time
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; i++)
    {
        // Calculate the 4D integral using the gauss_legendre_quadrature function
        result = gauss_legendre_quadrature(function_to_integrate, bounds);
    }

    // Stop measuring time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time) / 1000;

    std::cout << "Approximate 4D integral: " << result << std::endl;
    std::cout << "Time taken for computation: " << duration.count() << " microseconds" << std::endl;

    return 0;
}
