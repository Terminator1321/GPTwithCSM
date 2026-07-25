#include "Embedding.hpp"

#include <stdexcept>

Embedding::Embedding(size_t vocab_size, size_t dim)
    : weights_(Parameter::xavier({vocab_size, dim}))
{
}

Tensor Embedding::forward(const std::vector<int>& token_ids) const
{
    size_t L = token_ids.size();
    size_t D = dim();
    Tensor out({L, D});

    for (size_t s = 0; s < L; ++s) {
        int t = token_ids[s];
        if (t < 0 || static_cast<size_t>(t) >= vocab_size()) {
            throw std::out_of_range("Embedding::forward: token id out of range");
        }
        for (size_t j = 0; j < D; ++j) {
            out(s, j) = weights_.value(static_cast<size_t>(t), j);
        }
    }
    return out;
}

void Embedding::backward(const std::vector<int>& token_ids, const Tensor& grad_output)
{
    size_t L = token_ids.size();
    size_t D = dim();

    if (grad_output.shape().size() != 2 || grad_output.shape()[0] != L || grad_output.shape()[1] != D) {
        throw std::invalid_argument("Embedding::backward: grad_output shape must be [L, dim]");
    }

    for (size_t s = 0; s < L; ++s) {
        int t = token_ids[s];
        if (t < 0 || static_cast<size_t>(t) >= vocab_size()) {
            throw std::out_of_range("Embedding::backward: token id out of range");
        }
        for (size_t j = 0; j < D; ++j) {
            // Accumulate, not overwrite: a repeated token must sum gradients
            // from every position it appears at.
            weights_.grad(static_cast<size_t>(t), j) += grad_output(s, j);
        }
    }
}

std::vector<Parameter*> Embedding::parameters()
{
    return { &weights_ };
}

Tensor Embedding::row(int token_id) const
{
    if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_size()) {
        throw std::out_of_range("Embedding::row: token id out of range");
    }
    Tensor out({dim()});
    for (size_t j = 0; j < dim(); ++j) out(j) = weights_.value(static_cast<size_t>(token_id), j);
    return out;
}
