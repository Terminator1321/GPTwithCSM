#include "Transformer.hpp"

TransformerBlock::TransformerBlock(size_t embedDim, size_t numHeads) : ln1(embedDim), attention(embedDim, numHeads), ln2(embedDim), mlp(embedDim)
{
}

Tensor TransformerBlock::forward(const Tensor &x)
{
    Tensor input = x.clone();
    Tensor normed1 = ln1.forward(input);
    Tensor attnOut = attention.forward(normed1);
    Tensor residual1 = Tensor::add(input, attnOut);

    Tensor normed2 = ln2.forward(residual1);
    Tensor mlpOut = mlp.forward(normed2);
    Tensor output = Tensor::add(residual1, mlpOut);

    return output;
}
