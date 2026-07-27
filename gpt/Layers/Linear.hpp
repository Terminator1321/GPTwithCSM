#ifndef LINEAR_LAYER_HPP
#define LINEAR_LAYER_HPP

#include "../Tensor/Tensor.hpp"
#include "../helper/RandomWeights.hpp"

class LinearLayer {
public:
    LinearLayer(int input_size, int output_size = 1);

    Tensor forward(const Tensor& input) const;
    Tensor backward(const Tensor& output_gradients) const;

    Tensor weights;
    Tensor bias;
};

#endif
