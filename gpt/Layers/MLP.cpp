#include "MLP.hpp"

FeedForward::FeedForward(size_t embedDim) : fc1(embedDim, embedDim * 4), fc2(embedDim * 4, embedDim)
{
}

Tensor FeedForward::forward(const Tensor &x)
{
    Tensor out = fc1.forward(x);
    out = Activation::geluForward(out);
    out = fc2.forward(out);
    return out;
}