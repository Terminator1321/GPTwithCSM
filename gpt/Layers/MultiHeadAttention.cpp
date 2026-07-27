#include "MultiHeadAttention.hpp"

#include <stdexcept>
#include <vector>

MultiHeadAttention::MultiHeadAttention(size_t embedDim, size_t numHeads) : Wq(embedDim, embedDim), Wk(embedDim, embedDim), Wv(embedDim, embedDim), Wo(embedDim, embedDim), embedDim(embedDim), numHeads(numHeads), headDim(embedDim / numHeads)
{
    if (embedDim % numHeads != 0)
    {
        throw std::runtime_error(
            "embedDim must be divisible by numHeads.");
    }
}

Tensor MultiHeadAttention::forward(const Tensor &x)
{
    const size_t seqLen = x.rows();

    Tensor Q = Wq.forward(x);
    Tensor K = Wk.forward(x);
    Tensor V = Wv.forward(x);

    Q = Q.reshape({seqLen, numHeads, headDim}).permute({1, 0, 2});
    K = K.reshape({seqLen, numHeads, headDim}).permute({1, 0, 2});
    V = V.reshape({seqLen, numHeads, headDim}).permute({1, 0, 2});

    auto qHeads = Tensor::split(Q, 0, 1);
    auto kHeads = Tensor::split(K, 0, 1);
    auto vHeads = Tensor::split(V, 0, 1);

    std::vector<Tensor> outputs;

    for (size_t i = 0; i < numHeads; i++)
    {
        outputs.push_back(
            attention.forward(
                qHeads[i],
                kHeads[i],
                vHeads[i]));
    }

    Tensor out = Tensor::concatenate(outputs, 0);

    out = out.permute({1, 0, 2});
    out = out.flatten(1, 2);

    return Wo.forward(out);
}