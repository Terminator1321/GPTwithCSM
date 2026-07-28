#ifndef MLP_HPP
#define MLP_HPP

#include "Linear.hpp"
#include "../NameSpaces/ActivationFunction.hpp"
#include "../core/Module.hpp"

class FeedForward : public Module
{
private:
    LinearLayer fc1;
    LinearLayer fc2;

    Tensor fc1_out;

public:
    FeedForward(size_t embedDim);
    Tensor forward(Tensor &x);
    Tensor backward(Tensor &gradOutput);

    std::vector<Parameter*> parameters() override;
};

#endif
