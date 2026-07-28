#include "ScaledDotProductAttention.hpp"

#include "../helper/mask.hpp"
#include "../NameSpaces/ActivationFunction.hpp"

#include <cmath>

Tensor ScaledDotProductAttention::forward(const Tensor &Q, const Tensor &K, const Tensor &V)
{
    lastQ = Q.clone();
    lastK = K.clone();
    lastV = V.clone();

    const size_t ndim = K.shape().size();
    Tensor KT = K.transpose(ndim - 2, ndim - 1);
    Tensor scores = Tensor::matmul(Q, KT);
    const double dk = static_cast<double>(K.shape().back());

    scale = 1.0 / std::sqrt(dk);
    scores = Tensor::multiply(scores, scale);
    Mask::causalMask(scores);
    lastAttnWeights = Activation::softmaxForward(scores);
    Tensor output = Tensor::matmul(lastAttnWeights, V);

    return output;
}

ScaledDotProductAttention::Gradients ScaledDotProductAttention::backward(const Tensor &gradOutput)
{
    const size_t ndim = lastAttnWeights.shape().size();

    Tensor attnWeightsT = lastAttnWeights.transpose(ndim - 2, ndim - 1);
    Tensor dV = Tensor::matmul(attnWeightsT, gradOutput);
    Tensor VT = lastV.transpose(ndim - 2, ndim - 1);
    Tensor dAttnWeights = Tensor::matmul(gradOutput, VT);
    Tensor dScaledScores = Activation::softmaxBackward(dAttnWeights, lastAttnWeights);
    Tensor dRawScores = Tensor::multiply(dScaledScores, scale);
    Tensor dQ = Tensor::matmul(dRawScores, lastK);
    Tensor dRawScoresT = dRawScores.transpose(ndim - 2, ndim - 1);
    Tensor dK = Tensor::matmul(dRawScoresT, lastQ);

    return { dQ, dK, dV };
}
