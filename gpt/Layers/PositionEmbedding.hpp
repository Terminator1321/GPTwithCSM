#pragma once
#ifndef POSITIONEMBEDDING_HPP
#define POSITIONEMBEDDING_HPP
#include "../Tensor/Tensor.hpp"
#include "../core/Parameter.hpp"
#include "../core/Module.hpp"

class PositionEmbedding : public Module
{
private:
    Parameter embeddings;

    size_t maxSeqLen;
    size_t embedDim;
    size_t lastSeqLen = 0;

public:
    PositionEmbedding(size_t maxSeqLen, size_t embedDim);

    Tensor forward(size_t seqLen);
    void backward(const Tensor &gradOutput);

    std::vector<Parameter*> parameters() override;

    Tensor& getWeights();
    const Tensor& getWeights() const;
};
#endif
