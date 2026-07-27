#ifndef MLP_HPP
#define MLP_HPP

#include "Linear.hpp"
#include "../NameSpaces/ActivationFunction.hpp"

class FeedForward
{
private:
    LinearLayer fc1;
    LinearLayer fc2;

public:
    FeedForward(size_t embedDim);
    Tensor forward(const Tensor &x);
    Tensor backward(const Tensor& gradOutput);

    void zeroGrad();
};

#endif