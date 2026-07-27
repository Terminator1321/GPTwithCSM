#pragma once
#ifndef POSITIONEMBEDDING_HPP
#define POSITIONEMBEDDING_HPP
#include "../Tensor/Tensor.hpp"

class PositionEmbedding
{
private:
    Tensor embeddings;

    size_t maxSeqLen;
    size_t embedDim;

public:
    PositionEmbedding(size_t maxSeqLen, size_t embedDim);

    Tensor forward(size_t seqLen);

    Tensor& getWeights();
    const Tensor& getWeights() const;
};
#endif
