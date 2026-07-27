#ifndef LAYERNORMS_HPP
#define LAYERNORMS_HPP

#include "../Tensor/Tensor.hpp"
#include <vector>
#include <math.h>
class LayerNorms
{
    private:
        Tensor gamma;
        Tensor beta;
        Tensor normalized;

        double mean;
        double variance;
        const double EPS = 1e-5;
        
    public:
        LayerNorms(size_t embedding_dim);

        Tensor forward(Tensor &input);
        Tensor backward(Tensor &grad);


};

#endif