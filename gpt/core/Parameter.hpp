#pragma once

#include "../Tensor/Tensor.hpp"
#include <stdexcept>

class Parameter
{
public:
    Tensor value;
    Tensor grad;
    Tensor m;
    Tensor v;
    bool trainable;

    Parameter() : trainable(true) {}

    explicit Parameter(Tensor initial_value, bool trainable_ = true)
        : value(std::move(initial_value)),
          grad(value.shape()),
          m(value.shape()),
          v(value.shape()),
          trainable(trainable_)
    {
    }

    static Parameter zeros(const std::vector<size_t>& shape, bool trainable = true)
    {
        return Parameter(Tensor::zeros(shape), trainable);
    }

    static Parameter xavier(const std::vector<size_t>& shape, bool trainable = true)
    {
        return Parameter(Tensor::xavier(shape), trainable);
    }

    static Parameter he(const std::vector<size_t>& shape, bool trainable = true)
    {
        return Parameter(Tensor::he(shape), trainable);
    }

    void zero_grad()
    {
        grad.zero();
    }

    const std::vector<size_t>& shape() const { return value.shape(); }
    size_t size() const { return value.size(); }
};
