#include "Linear.hpp"

#include <stdexcept>
#include <vector>

LinearLayer::LinearLayer(int input_size, int output_size) : weights({static_cast<size_t>(output_size), static_cast<size_t>(input_size)}), bias({static_cast<size_t>(output_size)}) {
    if (input_size <= 0 || output_size <= 0) {
        throw std::invalid_argument("Layer sizes must be positive");
    }

    for (int out = 0; out < output_size; ++out) {
        for (int in = 0; in < input_size; ++in) {
            weights(out, in) = (in + out) % 2 == 0 ? 0.2 : -0.4;
        }
    }

    bias.fill(0.0);
}

Tensor LinearLayer::forward(const Tensor& input) const {
    if (input.shape().size() != 1 || input.size() != weights.cols()) {
        throw std::invalid_argument("Input must be a 1D tensor matching the layer input size");
    }

    Tensor output({static_cast<size_t>(weights.rows())});
    for (size_t out = 0; out < weights.rows(); ++out) {
        double value = bias(out);
        for (size_t in = 0; in < input.size(); ++in) {
            value += input(in) * weights(out, in);
        }
        output(out) = value;
    }
    return output;
}

Tensor LinearLayer::backward(const Tensor& output_gradients) const {
    if (output_gradients.shape().size() != 1 || output_gradients.size() != weights.rows()) {
        throw std::invalid_argument("Output gradient must be a 1D tensor matching the layer output size");
    }

    Tensor input_gradients({weights.cols()});
    for (size_t out = 0; out < weights.rows(); ++out) {
        for (size_t in = 0; in < weights.cols(); ++in) {
            input_gradients(in) += output_gradients(out) * weights(out, in);
        }
    }
    return input_gradients;
}
