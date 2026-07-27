#ifndef MULTIHEADATTENTION_HPP
#define MULTIHEADATTENTION_HPP

#include "Linear.hpp"
#include "ScaledDotProductAttention.hpp"
#include "../Tensor/Tensor.hpp"

class MultiHeadAttention
{
    private:
        LinearLayer Wq;
        LinearLayer Wk;
        LinearLayer Wv;
        LinearLayer Wo;

        ScaledDotProductAttention attention;

        size_t embedDim;
        size_t numHeads;
        size_t headDim;
    public:
        MultiHeadAttention(size_t embedDim, size_t numHeads);
        Tensor forward(const Tensor& x);
};

#endif