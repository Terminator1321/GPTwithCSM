#include "ScaledDotProductAttention.hpp"

#include "../helper/mask.hpp"
#include "../NameSpaces/ActivationFunction.hpp"

#include <cmath>

Tensor ScaledDotProductAttention::forward(const Tensor &Q, const Tensor &K, const Tensor &V)
{
    const size_t ndim = K.shape().size();
    Tensor KT = K.transpose(ndim - 2, ndim - 1);
    Tensor scores = Tensor::matmul(Q, KT);
    const double dk = static_cast<double>(K.shape().back());

    scores /= std::sqrt(dk);
    Mask::causalMask(scores);
    scores = Activation::softmaxForward(scores);
    Tensor output = Tensor::matmul(scores, V);

    return output;
}