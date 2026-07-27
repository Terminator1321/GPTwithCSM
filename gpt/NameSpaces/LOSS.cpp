#include "Loss.hpp"

#include <cmath>
#include <algorithm>

namespace Loss
{

    constexpr double EPS = 1e-12;

    double crossEntropyForward(const Tensor &probabilities, size_t targetIndex)
    {
        double p = probabilities.at({targetIndex});
        p = std::clamp(p, EPS, 1.0 - EPS);
        return -std::log(p);
    }

    Tensor crossEntropyBackward(const Tensor &probabilities, size_t targetIndex)
    {
        Tensor grad(probabilities.shape());
        for (size_t i = 0; i < probabilities.size(); i++)
            grad.at({i}) = 0.0;

        double p = std::clamp(probabilities.at({targetIndex}), EPS, 1.0 - EPS);
        grad.at({targetIndex}) = -1.0 / p;
        return grad;
    }

}