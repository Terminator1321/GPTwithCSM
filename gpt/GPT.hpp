#pragma once
#include <vector>
#include "Layers/Embedding.hpp"
#include "Layers/PositionEmbedding.hpp"
#include "Transformer.hpp"
#include "Layers/LayerNorms.hpp"
#include "Layers/Linear.hpp"
#include "Tensor/Tensor.hpp"
#include "core/Module.hpp"

class GPT : public Module
{
private:
    size_t vocabSize;
    size_t embedDim;
    size_t maxSeqLen;
    size_t numHeads;
    size_t numLayers;

    Embedding tokenEmbedding;
    PositionEmbedding positionEmbedding;
    std::vector<TransformerBlock> blocks;
    LayerNorms finalNorm;
    LinearLayer lmHead;

    std::vector<int> lastTokens;

public:
    GPT(
        size_t vocabSize,
        size_t embedDim,
        size_t maxSeqLen,
        size_t numHeads,
        size_t numLayers
    );

    Tensor forward(const std::vector<int>& tokens);

    void backward(const Tensor& dLogits);
    double trainOnBatch(const std::vector<int>& tokens, const std::vector<int>& targets);

    std::vector<int> generate(std::vector<int> tokens, size_t maxNewTokens,
                               float temperature = 1.0f, int top_k = 0);
    std::vector<Parameter*> parameters() override;

    void summary() const;
};
