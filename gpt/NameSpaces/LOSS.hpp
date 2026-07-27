#ifndef LOSS_HPP
#define LOSS_HPP

#include "../Tensor/Tensor.hpp"
#include <cstddef>

namespace Loss
{
    double crossEntropyForward(const Tensor& probabilities, size_t targetIndex);
    Tensor crossEntropyBackward(const Tensor& probabilities, size_t targetIndex);
}

#endif