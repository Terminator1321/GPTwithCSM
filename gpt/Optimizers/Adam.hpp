#pragma once
#include "Optimizer.hpp"

class Adam : public Optimizer
{
public:
    Adam(double lr = 1e-3, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8) : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps)
    {
    }

    using Optimizer::step; 
    void step(Parameter& param) override;

private:
    double lr_;
    double beta1_;
    double beta2_;
    double eps_;
};
