#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP

#include "Tensor/Tensor.hpp"
#include "Layers/LayerNorms.hpp"
#include "Layers/MultiHeadAttention.hpp"
#include "Layers/MLP.hpp"
#include "core/Module.hpp"

class TransformerBlock : public Module
{
private:
    LayerNorms ln1;
    MultiHeadAttention attention;
    LayerNorms ln2;
    FeedForward mlp;

public:
    TransformerBlock(size_t embedDim, size_t numHeads);
    Tensor forward(Tensor &x);
    Tensor backward(Tensor &gradOutput);

    std::vector<Parameter*> parameters() override;
};

#endif
