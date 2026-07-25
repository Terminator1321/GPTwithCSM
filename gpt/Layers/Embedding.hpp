#pragma once
#include "../Tensor/Tensor.hpp"
#include "../core/Module.hpp"
#include "../core/Parameter.hpp"
#include <vector>

class Embedding : public Module
{
public:
    Embedding(size_t vocab_size, size_t dim);
    Tensor forward(const std::vector<int>& token_ids) const;
    void backward(const std::vector<int>& token_ids, const Tensor& grad_output);

    std::vector<Parameter*> parameters() override;

    size_t vocab_size() const { return weights_.value.shape()[0]; }
    size_t dim() const { return weights_.value.shape()[1]; }

    Tensor row(int token_id) const;

private:
    Parameter weights_;
};
