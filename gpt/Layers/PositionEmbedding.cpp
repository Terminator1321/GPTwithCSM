#include "PositionEmbedding.hpp"
#include <stdexcept>

PositionEmbedding::PositionEmbedding(size_t maxSeqLen, size_t embedDim) : maxSeqLen(maxSeqLen), embedDim(embedDim), embeddings({maxSeqLen, embedDim})
{
    embeddings.random(0.0, 0.02);
}

Tensor PositionEmbedding::forward(size_t seqLen)
{
    if (seqLen > maxSeqLen)
    {
        throw std::runtime_error(
            "Sequence length exceeds maximum positional embeddings.");
    }

    // embedDim is taken in full; only the sequence-length axis is sliced,
    // so this returns a (seqLen, embedDim) view without copying data.
    return embeddings.slice(0, 0, seqLen);
}

Tensor &PositionEmbedding::getWeights()
{
    return embeddings;
}

const Tensor &PositionEmbedding::getWeights() const
{
    return embeddings;
}
