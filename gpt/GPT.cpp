#include "GPT.hpp"
#include "ModelSummary.hpp"
#include "NameSpaces/ActivationFunction.hpp"
#include "NameSpaces/LOSS.hpp"
#include "helper/Sampling.hpp"

#include <algorithm>
#include <stdexcept>

GPT::GPT(size_t vocabSize, size_t embedDim, size_t maxSeqLen, size_t numHeads, size_t numLayers) : vocabSize(vocabSize),  embedDim(embedDim),  maxSeqLen(maxSeqLen),  numHeads(numHeads),  numLayers(numLayers),  tokenEmbedding(vocabSize, embedDim),  positionEmbedding(maxSeqLen, embedDim),  finalNorm(embedDim),  lmHead(embedDim, vocabSize)
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

    lastTokens = tokens;

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

void GPT::backward(const Tensor &dLogits)
{
    if (lastTokens.empty())
    {
        throw std::runtime_error("GPT::backward: called before forward()");
    }

    Tensor dX = dLogits.clone();
    dX = lmHead.backward(dX);
    dX = finalNorm.backward(dX);

    for (auto it = blocks.rbegin(); it != blocks.rend(); ++it)
    {
        dX = it->backward(dX);
    }

    positionEmbedding.backward(dX);
    tokenEmbedding.backward(lastTokens, dX);
}

double GPT::trainOnBatch(const std::vector<int> &tokens, const std::vector<int> &targets)
{
    if (tokens.size() != targets.size())
    {
        throw std::invalid_argument("GPT::trainOnBatch: tokens and targets must be the same length");
    }

    Tensor logits = forward(tokens);
    const size_t seqLen = tokens.size();

    Tensor dLogits({seqLen, vocabSize});
    double totalLoss = 0.0;

    for (size_t s = 0; s < seqLen; s++)
    {
        Tensor row({vocabSize});
        for (size_t j = 0; j < vocabSize; j++)
            row(j) = logits(s, j);

        Tensor probs = Activation::softmaxForward(row);
        totalLoss += Loss::crossEntropyForward(probs, static_cast<size_t>(targets[s]));

        for (size_t j = 0; j < vocabSize; j++)
            dLogits(s, j) = probs(j);
        dLogits(s, static_cast<size_t>(targets[s])) -= 1.0;
    }

    totalLoss /= static_cast<double>(seqLen);
    dLogits = Tensor::multiply(dLogits, 1.0 / static_cast<double>(seqLen));

    backward(dLogits);
    return totalLoss;
}

void GPT::summary() const
{
    ModelSummary::print(GPTConfig{vocabSize, embedDim, maxSeqLen, numHeads, numLayers});
}

std::vector<Parameter*> GPT::parameters()
{
    std::vector<Parameter*> params;

    auto append = [&params](std::vector<Parameter*> sub) {
        params.insert(params.end(), sub.begin(), sub.end());
    };

    append(tokenEmbedding.parameters());
    append(positionEmbedding.parameters());
    for (auto &block : blocks)
        append(block.parameters());
    append(finalNorm.parameters());
    append(lmHead.parameters());

    return params;
}

std::vector<int> GPT::generate(std::vector<int> tokens, size_t maxNewTokens, float temperature, int top_k)
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

        int nextToken = sampleNextToken(lastLogits, temperature, top_k);

        tokens.push_back(nextToken);
    }

    return tokens;
}
