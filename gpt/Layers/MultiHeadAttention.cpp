#include "MultiHeadAttention.hpp"

#include <stdexcept>
#include <vector>

MultiHeadAttention::MultiHeadAttention(size_t embedDim, size_t numHeads)
    : Wq(embedDim, embedDim), Wk(embedDim, embedDim), Wv(embedDim, embedDim), Wo(embedDim, embedDim),
      embedDim(embedDim), numHeads(numHeads), headDim(embedDim / numHeads)
{
    if (embedDim % numHeads != 0)
    {
        throw std::runtime_error(
            "embedDim must be divisible by numHeads.");
    }
    heads.resize(numHeads);
}

Tensor MultiHeadAttention::forward(Tensor &x)
{
    const size_t seqLen = x.rows();
    lastSeqLen = seqLen;

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
            heads[i].forward(
                qHeads[i],
                kHeads[i],
                vHeads[i]));
    }

    Tensor out = Tensor::concatenate(outputs, 0);

    out = out.permute({1, 0, 2});
    out = out.flatten(1, 2);

    return Wo.forward(out);
}

Tensor MultiHeadAttention::backward(Tensor &gradOutput)
{
    const size_t seqLen = lastSeqLen;

    Tensor dConcatFlat = Wo.backward(gradOutput);
    Tensor dConcatHeads = dConcatFlat.reshape({seqLen, numHeads, headDim}).permute({1, 0, 2});
    auto dHeadsSplit = Tensor::split(dConcatHeads, 0, 1);

    std::vector<Tensor> dQParts, dKParts, dVParts;
    dQParts.reserve(numHeads);
    dKParts.reserve(numHeads);
    dVParts.reserve(numHeads);

    for (size_t i = 0; i < numHeads; i++)
    {
        auto g = heads[i].backward(dHeadsSplit[i]);
        dQParts.push_back(std::move(g.dQ));
        dKParts.push_back(std::move(g.dK));
        dVParts.push_back(std::move(g.dV));
    }

    Tensor dQHeads = Tensor::concatenate(dQParts, 0); 
    Tensor dKHeads = Tensor::concatenate(dKParts, 0);
    Tensor dVHeads = Tensor::concatenate(dVParts, 0);
    Tensor dQFlat = dQHeads.permute({1, 0, 2}).reshape({seqLen, embedDim});
    Tensor dKFlat = dKHeads.permute({1, 0, 2}).reshape({seqLen, embedDim});
    Tensor dVFlat = dVHeads.permute({1, 0, 2}).reshape({seqLen, embedDim});
    Tensor dXq = Wq.backward(dQFlat);
    Tensor dXk = Wk.backward(dKFlat);
    Tensor dXv = Wv.backward(dVFlat);
    Tensor dX = Tensor::add(Tensor::add(dXq, dXk), dXv);
    return dX;
}

std::vector<Parameter*> MultiHeadAttention::parameters()
{
    std::vector<Parameter*> params;
    for (LinearLayer* layer : { &Wq, &Wk, &Wv, &Wo })
    {
        auto layerParams = layer->parameters();
        params.insert(params.end(), layerParams.begin(), layerParams.end());
    }
    return params;
}
