#ifndef ACTIVATIONFUNCTION_HPP
#define ACTIVATIONFUNCTION_HPP

#include "../Tensor/Tensor.hpp"

namespace Activation
{
    Tensor softmaxForward(const Tensor& input);
    Tensor softmaxBackward(const Tensor& gradOutput, const Tensor& softmaxOutput);
    Tensor geluForward(const Tensor& input);
    Tensor geluBackward(const Tensor& gradOutput, const Tensor& input);
}

#endif