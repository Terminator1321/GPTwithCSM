#include "LayerNorms.hpp"

LayerNorms::LayerNorms(size_t embedding_dim) : gamma(std::vector<size_t>{embedding_dim}), beta(std::vector<size_t>{embedding_dim})
{
    gamma.fill(1.0);
    beta.fill(0.0);
}

Tensor LayerNorms::forward(Tensor &input)
{
    Tensor output = input.clone();
    size_t seq_len = input.rows();
    size_t embedding_dim = input.cols();

    for (size_t token = 0; token < seq_len; token++)
    {
        mean = 0.0;
        for (size_t emb = 0; emb < embedding_dim; emb++)
        {
            mean += input(token, emb);
        }
        mean /= embedding_dim;

        variance = 0.0;
        for (size_t emb = 0; emb < embedding_dim; emb++)
        {
            double diff = input(token, emb) - mean;
            variance += diff * diff;
        }

        variance /= embedding_dim;
        double std = std::sqrt(variance + EPS);

        for (size_t emb = 0; emb < embedding_dim; emb++)
        {
            double normalized = (input(token, emb) - mean) / std;
            output(token, emb) = gamma(emb) * normalized + beta(emb);
        }
    }

    return output;
}