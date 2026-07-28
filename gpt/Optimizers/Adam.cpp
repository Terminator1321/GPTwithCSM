#include "Adam.hpp"
#include <cmath>

void Adam::step(Parameter &param)
{
    if (!param.trainable)
        return;

    const double t = static_cast<double>(timestep_ + 1);
    const double biasCorrection1 = 1.0 - std::pow(beta1_, t);
    const double biasCorrection2 = 1.0 - std::pow(beta2_, t);

    const size_t n = param.value.size();
    for (size_t i = 0; i < n; i++)
    {
        double g = param.grad(i);

        double m = beta1_ * param.m(i) + (1.0 - beta1_) * g;
        double v = beta2_ * param.v(i) + (1.0 - beta2_) * g * g;

        param.m(i) = m;
        param.v(i) = v;

        double mHat = m / biasCorrection1;
        double vHat = v / biasCorrection2;

        param.value(i) -= lr_ * mHat / (std::sqrt(vHat) + eps_);
    }
}
