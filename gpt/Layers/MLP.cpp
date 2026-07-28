#include "MLP.hpp"

FeedForward::FeedForward(size_t embedDim) : fc1(embedDim, embedDim * 4), fc2(embedDim * 4, embedDim)
{
}

Tensor FeedForward::forward(Tensor &x)
{
    Tensor out = fc1.forward(x);
    fc1_out = out.clone();
    out = Activation::geluForward(out);
    out = fc2.forward(out);
    return out;
}

Tensor FeedForward::backward(Tensor &gradOutput)
{
    Tensor dFc2Input = fc2.backward(gradOutput);
    Tensor dGelu = Activation::geluBackward(dFc2Input, fc1_out);
    Tensor dX = fc1.backward(dGelu);
    return dX;
}

std::vector<Parameter*> FeedForward::parameters()
{
    std::vector<Parameter*> params;
    auto p1 = fc1.parameters();
    auto p2 = fc2.parameters();
    params.insert(params.end(), p1.begin(), p1.end());
    params.insert(params.end(), p2.begin(), p2.end());
    return params;
}
