#ifndef LAYERNORMS_HPP
#define LAYERNORMS_HPP

#include "../Tensor/Tensor.hpp"
#include "../core/Parameter.hpp"
#include "../core/Module.hpp"
#include <vector>
#include <math.h>

class LayerNorms : public Module
{
    private:
        Tensor last_input;
        Tensor normalized;
        Tensor inv_std;
        const double EPS = 1e-5;

    public:
        Parameter gamma;
        Parameter beta;

        LayerNorms(size_t embedding_dim);

        Tensor forward(Tensor &input);
        Tensor backward(Tensor &grad);

        std::vector<Parameter*> parameters() override;
};

#endif
