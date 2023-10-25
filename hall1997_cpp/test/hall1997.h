#ifndef HALL1997_H
#define HALL1997_H

enum sign
{
    upper = 1,
    lower = -1
};

double fpt_density(double t, double mu, double a, double b, double x0, sign bdy, int trunc_num = 100);
double trans_density(double x, double mu, double a, double b, double x0, double T, int trunc_num = 100);

#endif // HALL1997_H
