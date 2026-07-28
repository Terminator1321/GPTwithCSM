#ifndef LINEAR_LAYER_HPP
#define LINEAR_LAYER_HPP

#include "../Tensor/Tensor.hpp"
#include "../helper/RandomWeights.hpp"
#include "../core/Parameter.hpp"
#include "../core/Module.hpp"
#include <vector>

class LinearLayer : public Module {
private:
    Tensor last_input;
public:
    LinearLayer(int input_size, int output_size = 1);

    Tensor forward(Tensor& input);
    Tensor backward(Tensor& output_gradients);

    std::vector<Parameter*> parameters() override;

    Parameter weights;
    Parameter bias;
};

#endif
