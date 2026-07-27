#pragma once
#include <vector>
#include "Layers/Embedding.hpp"
#include "Layers/PositionEmbedding.hpp"
#include "Transformer.hpp"
#include "Layers/LayerNorms.hpp"
#include "Layers/Linear.hpp"
#include "Tensor/Tensor.hpp"

class GPT
{
private:
    size_t vocabSize;
    size_t embedDim;
    size_t maxSeqLen;
    size_t numLayers;

    Embedding tokenEmbedding;
    PositionEmbedding positionEmbedding;
    std::vector<TransformerBlock> blocks;
    LayerNorms finalNorm;
    LinearLayer lmHead;

public:
    GPT(
        size_t vocabSize,
        size_t embedDim,
        size_t maxSeqLen,
        size_t numHeads,
        size_t numLayers
    );

    Tensor forward(const std::vector<int>& tokens);
    std::vector<int> generate(std::vector<int> tokens, size_t maxNewTokens);
};
