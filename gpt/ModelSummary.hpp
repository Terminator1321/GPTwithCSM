#pragma once
#include <cstddef>

// Plain-data snapshot of the hyperparameters GPT is constructed with.
// Kept separate from the GPT class so the summary printer has no
// dependency on Tensor/Module internals -- it only needs the five
// numbers that determine every shape and parameter count in the model.
struct GPTConfig
{
    size_t vocabSize;
    size_t embedDim;
    size_t maxSeqLen;
    size_t numHeads;
    size_t numLayers;
};

namespace ModelSummary
{
    // Prints a PyTorch/TensorFlow-style architecture table for a GPT
    // built from `cfg` to stdout, using Unicode box-drawing characters.
    void print(const GPTConfig &cfg);
}
