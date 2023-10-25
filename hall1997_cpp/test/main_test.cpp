#include "quadrature.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

// Function to integrate (modify this function as needed)
double function_to_integrate(double x, double y, double z, double w)
{
    // Replace this with the function you want to integrate
    return x * x + y * y + y * z + exp(-z) * cos(w); // Example: integrating x^2 + y^2 + y*z + exp(-z) * cos(w)
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