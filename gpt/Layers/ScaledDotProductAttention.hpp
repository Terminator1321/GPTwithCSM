#ifndef SCALEDDOTPRODUCTATTENTION_HPP
#define SCALEDDOTPRODUCTATTENTION_HPP

#include "../Tensor/Tensor.hpp"

class ScaledDotProductAttention
{
public:
    struct Gradients
    {
        Tensor dQ;
        Tensor dK;
        Tensor dV;
    };

    Tensor forward(const Tensor &Q, const Tensor &K, const Tensor &V);
    Gradients backward(const Tensor &gradOutput);

private:
    Tensor lastQ;
    Tensor lastK;
    Tensor lastV;
    Tensor lastAttnWeights; 
    double scale = 1.0;
};

#endif
