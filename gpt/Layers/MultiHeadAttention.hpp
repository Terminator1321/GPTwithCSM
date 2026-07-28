#ifndef MULTIHEADATTENTION_HPP
#define MULTIHEADATTENTION_HPP

#include "Linear.hpp"
#include "ScaledDotProductAttention.hpp"
#include "../Tensor/Tensor.hpp"
#include "../core/Module.hpp"
#include <vector>

class MultiHeadAttention : public Module
{
    private:
        LinearLayer Wq;
        LinearLayer Wk;
        LinearLayer Wv;
        LinearLayer Wo;

        std::vector<ScaledDotProductAttention> heads;

        size_t embedDim;
        size_t numHeads;
        size_t headDim;
        size_t lastSeqLen = 0;

    public:
        MultiHeadAttention(size_t embedDim, size_t numHeads);
        Tensor forward(Tensor &x);
        Tensor backward(Tensor &gradOutput);

        std::vector<Parameter*> parameters() override;
};

#endif
