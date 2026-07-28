#pragma once

#include <cstddef>
#include <string>

#include "GPT.hpp"

// Custom binary checkpoint format (.csm = "Checkpoint State Model").
//
// Layout:
//   char[4]   magic            "CSM1"
//   uint32    version
//   uint64    vocabSize
//   uint64    embedDim
//   uint64    maxSeqLen
//   uint64    numHeads
//   uint64    numLayers
//   uint64    globalStep       (training step this checkpoint was taken at)
//   uint64    paramCount
//   per parameter:
//     uint64        ndim
//     uint64[ndim]  shape
//     uint64        elementCount
//     double[count] value      (weights)
//     double[count] m          (Adam 1st moment)
//     double[count] v          (Adam 2nd moment)
//
// The architecture fields are used to sanity-check that a checkpoint is
// being loaded into a model with a matching configuration.

// Saves the full model state (weights + optimizer moments) to `path`.
// Returns true on success.
bool saveCheckpoint(const std::string& path, GPT& gpt, size_t globalStep, size_t vocabSize, size_t embedDim, size_t maxSeqLen, size_t numHeads, size_t numLayers);
bool loadCheckpoint(const std::string& path, GPT& gpt, size_t& globalStep, size_t vocabSize, size_t embedDim, size_t maxSeqLen, size_t numHeads, size_t numLayers);
