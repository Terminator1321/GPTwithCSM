#include "Transformer.hpp"

TransformerBlock::TransformerBlock(size_t embedDim, size_t numHeads) : ln1(embedDim), attention(embedDim, numHeads), ln2(embedDim), mlp(embedDim)
{
}

Tensor TransformerBlock::forward(Tensor &x)
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

Tensor TransformerBlock::backward(Tensor &gradOutput)
{
    Tensor dMlpOut = gradOutput;
    Tensor dResidual1 = gradOutput.clone();

    Tensor dNormed2 = mlp.backward(dMlpOut);
    Tensor dResidual1FromLn2 = ln2.backward(dNormed2);
    dResidual1 = Tensor::add(dResidual1, dResidual1FromLn2);

    Tensor dAttnOut = dResidual1;
    Tensor dInput = dResidual1.clone();

    Tensor dNormed1 = attention.backward(dAttnOut);
    Tensor dInputFromLn1 = ln1.backward(dNormed1);
    dInput = Tensor::add(dInput, dInputFromLn1);

    return dInput;
}

std::vector<Parameter*> TransformerBlock::parameters()
{
    std::vector<Parameter*> params;
    for (auto* p : { &ln1, &ln2 })
    {
        auto sub = p->parameters();
        params.insert(params.end(), sub.begin(), sub.end());
    }
    auto attnParams = attention.parameters();
    params.insert(params.end(), attnParams.begin(), attnParams.end());
    auto mlpParams = mlp.parameters();
    params.insert(params.end(), mlpParams.begin(), mlpParams.end());
    return params;
}
