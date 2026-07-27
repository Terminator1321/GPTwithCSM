#include "GPT.hpp"

#include <algorithm>
#include <stdexcept>

GPT::GPT(size_t vocabSize, size_t embedDim, size_t maxSeqLen, size_t numHeads, size_t numLayers) : vocabSize(vocabSize),  embedDim(embedDim),  maxSeqLen(maxSeqLen),  numLayers(numLayers),  tokenEmbedding(vocabSize, embedDim),  positionEmbedding(maxSeqLen, embedDim),  finalNorm(embedDim),  lmHead(embedDim, vocabSize)
{
    blocks.reserve(numLayers);
    for (size_t i = 0; i < numLayers; i++)
    {
        blocks.emplace_back(embedDim, numHeads);
    }
}

Tensor GPT::forward(const std::vector<int> &tokens)
{
    if (tokens.empty())
    {
        throw std::invalid_argument("GPT::forward: tokens must not be empty");
    }
    if (tokens.size() > maxSeqLen)
    {
        throw std::runtime_error("GPT::forward: sequence length exceeds maxSeqLen");
    }

    Tensor tokenEmb = tokenEmbedding.forward(tokens);
    Tensor posEmb = positionEmbedding.forward(tokens.size());
    Tensor x = Tensor::add(tokenEmb, posEmb);

    for (auto &block : blocks)
    {
        x = block.forward(x);
    }

    x = finalNorm.forward(x);
    return lmHead.forward(x);
}

std::vector<int> GPT::generate(std::vector<int> tokens, size_t maxNewTokens)
{
    if (tokens.empty())
    {
        throw std::invalid_argument("GPT::generate: tokens must not be empty");
    }

    for (size_t step = 0; step < maxNewTokens; step++)
    {
        std::vector<int> context = tokens;
        if (context.size() > maxSeqLen)
        {
            context.erase(context.begin(), context.begin() + (context.size() - maxSeqLen));
        }

        Tensor logits = forward(context);
        const size_t seqLen = logits.shape()[0];
        Tensor lastLogits = logits.slice(0, seqLen - 1, seqLen).reshape({vocabSize});
        Tensor best = lastLogits.argmax();
        int nextToken = static_cast<int>(best(0));

        tokens.push_back(nextToken);
    }

    return tokens;
}
