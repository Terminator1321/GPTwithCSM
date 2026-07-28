#include "LayerNorms.hpp"

LayerNorms::LayerNorms(size_t embedding_dim)
    : gamma(Tensor({embedding_dim})), beta(Tensor({embedding_dim}))
{
    gamma.value.fill(1.0);
    beta.value.fill(0.0);
}

Tensor LayerNorms::forward(Tensor &input)
{
    last_input = input.clone();

    size_t seq_len = input.rows();
    size_t embedding_dim = input.cols();

    Tensor output = input.clone();
    normalized = Tensor({seq_len, embedding_dim});
    inv_std = Tensor({seq_len});

    for (size_t token = 0; token < seq_len; token++)
    {
        double mean = 0.0;
        for (size_t emb = 0; emb < embedding_dim; emb++)
            mean += input(token, emb);
        mean /= embedding_dim;

        double variance = 0.0;
        for (size_t emb = 0; emb < embedding_dim; emb++)
        {
            double diff = input(token, emb) - mean;
            variance += diff * diff;
        }
        variance /= embedding_dim;

        double std = std::sqrt(variance + EPS);
        inv_std(token) = 1.0 / std;

        for (size_t emb = 0; emb < embedding_dim; emb++)
        {
            double n = (input(token, emb) - mean) / std;
            normalized(token, emb) = n;
            output(token, emb) = gamma.value(emb) * n + beta.value(emb);
        }
    }

    return output;
}

Tensor LayerNorms::backward(Tensor &grad)
{
    size_t seq_len = last_input.rows();
    size_t embedding_dim = last_input.cols();

    Tensor input_grad({seq_len, embedding_dim});

    for (size_t token = 0; token < seq_len; token++)
    {
        double dnorm_sum = 0.0;
        double dnorm_dot_norm = 0.0;

        for (size_t emb = 0; emb < embedding_dim; emb++)
        {
            double dOut = grad(token, emb);
            beta.grad(emb) += dOut;
            gamma.grad(emb) += dOut * normalized(token, emb);

            double dNorm = dOut * gamma.value(emb);
            dnorm_sum += dNorm;
            dnorm_dot_norm += dNorm * normalized(token, emb);
        }

        double n = static_cast<double>(embedding_dim);
        double invStdTok = inv_std(token);

        for (size_t emb = 0; emb < embedding_dim; emb++)
        {
            double dOut = grad(token, emb);
            double dNorm = dOut * gamma.value(emb);

            double dx = invStdTok * (dNorm - dnorm_sum / n - normalized(token, emb) * dnorm_dot_norm / n);
            input_grad(token, emb) = dx;
        }
    }

    return input_grad;
}

std::vector<Parameter*> LayerNorms::parameters()
{
    return { &gamma, &beta };
}
