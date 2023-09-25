#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>

enum sign
{
    upper = 1,
    lower = -1
};

double fpt_density(double t, double mu, double a, double b, double x0, sign bdy, int trunc_num = 100)
{
    double tau = bdy * mu + b;
    double a1 = a - bdy * x0;
    double factor = (pow(t, -1.5) * exp(a1 * tau - 0.5 * pow(tau, 2) * t - b / (2 * a) * pow(a1, 2)) / sqrt(2 * M_PI));
    double result = 0;
    for (int j = 0; j < trunc_num; j++)
    {
        double rj = (2 * j + 1) * a - bdy * pow(-1, j) * x0;
        double term = pow(-1, j) * rj * exp(0.5 * (b / a - 1 / t) * pow(rj, 2));
        if (fabs(term) < 1e-20)
        {
            break;
        }
        result += term;
    }
    return result * factor;
}

double trans_density(double x, double mu, double a, double b, double x0, double T, int trunc_num = 100)
{
    x = x - x0;
    double factor = exp(mu * x - 0.5 * mu * mu * T) / sqrt(T);
    double result = exp(-0.5 * x * x / T) / sqrt(2.0 * M_PI);
    for (int j = 1; j < trunc_num; ++j)
    {
        double t1 = 4.0 * b * j * (2.0 * a * j + x0) - (x - 4.0 * a * j) * (x - 4.0 * a * j) / (2.0 * T);
        double t2 = 4.0 * b * j * (2.0 * a * j - x0) - (x + 4.0 * a * j) * (x + 4.0 * a * j) / (2.0 * T);
        double t3 = 2.0 * b * (2.0 * j - 1) * (2.0 * a * j - a + x0) - (x + (4.0 * j - 2) * a + 2.0 * x0) * (x + (4.0 * j - 2) * a + 2.0 * x0) / (2.0 * T);
        double t4 = 2.0 * b * (2.0 * j - 1) * (2.0 * a * j - a - x0) - (x - (4.0 * j - 2) * a + 2.0 * x0) * (x - (4.0 * j - 2) * a + 2.0 * x0) / (2.0 * T);
        double term = exp(t1) + exp(t2) - exp(t3) - exp(t4);
        if (fabs(term) < 1e-20)
        {
            break;
        }
        result += term / sqrt(2.0 * M_PI);
    }
    return result * factor;
}

int main()
{
    double mu = 1.0;
    double a = 1.5;
    double b = 0.3;
    double x0 = -0.1;

    double t = 1.0;
    double x = 0.0;
    double result1, result2, result3;

    auto start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
    {
        result1 = fpt_density(t, mu, a, b, x0, upper);
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    auto elapsed_time1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1);

    auto start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
    {
        result2 = fpt_density(t, mu, a, b, x0, lower);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    auto elapsed_time2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2);

    auto start3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
    {
        result3 = trans_density(x, mu, a, b, x0, t);
    }
    auto end3 = std::chrono::high_resolution_clock::now();
    auto elapsed_time3 = std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - start3);

    std::cout << "Result: " << result1 << " " << result2 << " " << result3 << std::endl;
    std::cout << "Execution Time: " << elapsed_time1.count() / 1000 << " " << elapsed_time2.count() / 1000 << " " << elapsed_time3.count() / 1000 << " nanoseconds" << std::endl;

    return 0;
}