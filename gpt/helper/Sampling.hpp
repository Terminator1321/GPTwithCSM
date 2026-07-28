#ifndef SAMPLING_HPP
#define SAMPLING_HPP

#include "../Tensor/Tensor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

// Samples the next token id from a 1D logits tensor of shape [vocabSize].
//
// temperature:
//   logits are divided by temperature before softmax.
//   temperature = 1.0 -> unchanged distribution (default)
//   temperature < 1.0 -> sharper / more deterministic
//   temperature > 1.0 -> flatter / more random
//
// top_k:
//   0 (default) -> disabled, full distribution is sampled
//   >0 -> only the k highest-logit tokens are kept, everything else is
//         set to -infinity before softmax so it can never be sampled.
inline int sampleNextToken(const Tensor &logits, float temperature = 1.0f, int top_k = 0)
{
    static std::mt19937 gen(std::random_device{}());

    const size_t vocabSize = logits.size();
    std::vector<double> scaled(vocabSize);

    // 1. Temperature scaling
    const double temp = (temperature <= 0.0f) ? 1e-6 : static_cast<double>(temperature);
    for (size_t i = 0; i < vocabSize; ++i)
        scaled[i] = logits(i) / temp;

    // 2. Top-K filtering: keep only the k largest logits, mask the rest to -inf
    if (top_k > 0 && static_cast<size_t>(top_k) < vocabSize)
    {
        std::vector<size_t> idx(vocabSize);
        for (size_t i = 0; i < vocabSize; ++i)
            idx[i] = i;

        std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                           [&](size_t a, size_t b)
                           { return scaled[a] > scaled[b]; });

        std::vector<bool> keep(vocabSize, false);
        for (int i = 0; i < top_k; ++i)
            keep[idx[i]] = true;

        for (size_t i = 0; i < vocabSize; ++i)
            if (!keep[i])
                scaled[i] = -std::numeric_limits<double>::infinity();
    }

    // 3. Numerically stable softmax over the (possibly filtered) logits
    double maxVal = *std::max_element(scaled.begin(), scaled.end());
    std::vector<double> probs(vocabSize);
    double sum = 0.0;
    for (size_t i = 0; i < vocabSize; ++i)
    {
        probs[i] = std::exp(scaled[i] - maxVal); // exp(-inf - maxVal) safely underflows to 0
        sum += probs[i];
    }
    for (auto &p : probs)
        p /= sum;

    // 4. Sample from the resulting categorical distribution
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(gen);
}

#endif
