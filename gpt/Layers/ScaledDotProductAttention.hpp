#ifndef SCALEDDOTPRODUCTATTENTION_HPP
#define SCALEDDOTPRODUCTATTENTION_HPP

#include "../Tensor/Tensor.hpp"

class ScaledDotProductAttention
{
public:
    Tensor forward(const Tensor &Q, const Tensor &K, const Tensor &V);
};

#endif