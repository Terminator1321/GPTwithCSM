#include "PositionEmbedding.hpp"
#include <stdexcept>

PositionEmbedding::PositionEmbedding(size_t maxSeqLen, size_t embedDim)
    : embeddings(Tensor({maxSeqLen, embedDim})), maxSeqLen(maxSeqLen), embedDim(embedDim)
{
    embeddings.value.random(0.0, 0.02);
}

Tensor PositionEmbedding::forward(size_t seqLen)
{
    if (seqLen > maxSeqLen)
    {
        throw std::runtime_error(
            "Sequence length exceeds maximum positional embeddings.");
    }

    lastSeqLen = seqLen;

    return embeddings.value.slice(0, 0, seqLen);
}

void PositionEmbedding::backward(const Tensor &gradOutput)
{
    const size_t seqLen = lastSeqLen;

    if (gradOutput.shape().size() != 2 || gradOutput.shape()[0] != seqLen || gradOutput.shape()[1] != embedDim)
    {
        throw std::invalid_argument("PositionEmbedding::backward: gradOutput shape must be [seqLen, embedDim]");
    }

    for (size_t pos = 0; pos < seqLen; pos++)
    {
        for (size_t j = 0; j < embedDim; j++)
        {
            embeddings.grad(pos, j) += gradOutput(pos, j);
        }
    }
}

std::vector<Parameter*> PositionEmbedding::parameters()
{
    return { &embeddings };
}

Tensor &PositionEmbedding::getWeights()
{
    return embeddings.value;
}

const Tensor &PositionEmbedding::getWeights() const
{
    return embeddings.value;
}
