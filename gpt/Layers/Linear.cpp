#include "Linear.hpp"
#include "../Tensor/TensorCuda.cuh"

#include <stdexcept>
#include <vector>

namespace
{
    // GPU dispatch/copy overhead only pays off on the bigger matmuls (MLP's
    // 4x expansion, the vocab-sized LM head, ...). Small layers/short
    // sequences stay on the CPU path, which is faster for tiny work.
    constexpr size_t kCudaLinearThreshold = 1u << 16;
}

LinearLayer::LinearLayer(int input_size, int output_size)
    : weights(Tensor::xavier({static_cast<size_t>(output_size), static_cast<size_t>(input_size)})),
      bias(Tensor({static_cast<size_t>(output_size)}))
{
    if (input_size <= 0 || output_size <= 0)
    {
        throw std::invalid_argument("Layer sizes must be positive");
    }

    // Previously initialized with a flat uniform(-0.05, 0.05) range regardless
    // of layer size (see helper/RandomWeights.hpp::get_weight), inconsistent
    // with Embedding/PositionEmbedding's fan-in/out-scaled init. Now uses the
    // same Xavier initialization as the rest of the network.
    bias.value.fill(0.0);
}

Tensor LinearLayer::forward(Tensor &input)
{
    last_input = input.clone();

    if (input.shape().size() == 2)
    {
        const size_t seqLen = input.shape()[0];
        const size_t inputSize = input.shape()[1];
        const size_t outputSize = weights.value.rows();

        if (inputSize != weights.value.cols())
        {
            throw std::invalid_argument("Input must be a 2D tensor whose last dimension matches the layer input size");
        }

        Tensor output({seqLen, outputSize});

#ifdef USE_CUDA
        if (seqLen * inputSize * outputSize >= kCudaLinearThreshold)
        {
            // input was just cloned into last_input, which clone() always
            // returns contiguous, so a raw pointer over its data is safe.
            cuda_linear_forward(last_input.data().data(), weights.value.data().data(),
                                 bias.value.data().data(), output.data().data(),
                                 seqLen, inputSize, outputSize);
            return output;
        }
#endif

        for (size_t token = 0; token < seqLen; ++token)
        {
            for (size_t out = 0; out < outputSize; ++out)
            {
                double value = bias.value(out);
                for (size_t in = 0; in < inputSize; ++in)
                {
                    value += input(token, in) * weights.value(out, in);
                }
                output(token, out) = value;
            }
        }
        return output;
    }

    if (input.shape().size() != 1 || input.size() != weights.value.cols())
    {
        throw std::invalid_argument("Input must be a 1D tensor matching the layer input size, or a 2D (seqLen, input_size) tensor");
    }

    const size_t outputSize = weights.value.rows();
    Tensor output({outputSize});
    for (size_t out = 0; out < outputSize; ++out)
    {
        double value = bias.value(out);
        for (size_t in = 0; in < input.size(); ++in)
        {
            value += input(in) * weights.value(out, in);
        }
        output(out) = value;
    }
    return output;
}

Tensor LinearLayer::backward(Tensor &output_gradients)
{
    const size_t outputSize = weights.value.rows();
    const size_t inputSize = weights.value.cols();

    if (last_input.shape().size() == 2)
    {
        const size_t seqLen = last_input.shape()[0];

        if (output_gradients.shape().size() != 2 ||
            output_gradients.shape()[0] != seqLen ||
            output_gradients.shape()[1] != outputSize)
        {
            throw std::invalid_argument("Output gradient must be a (seqLen, output_size) tensor matching the cached input");
        }

        Tensor input_gradients({seqLen, inputSize});

#ifdef USE_CUDA
        if (seqLen * inputSize * outputSize >= kCudaLinearThreshold)
        {
            Tensor dOut_contig = output_gradients.contiguous();
            cuda_linear_backward(dOut_contig.data().data(), last_input.data().data(),
                                  weights.value.data().data(), weights.grad.data().data(),
                                  bias.grad.data().data(), input_gradients.data().data(),
                                  seqLen, inputSize, outputSize);
            return input_gradients;
        }
#endif

        for (size_t token = 0; token < seqLen; ++token)
        {
            for (size_t out = 0; out < outputSize; ++out)
            {
                double dOut = output_gradients(token, out);
                bias.grad(out) += dOut;

                for (size_t in = 0; in < inputSize; ++in)
                {
                    weights.grad(out, in) += dOut * last_input(token, in);
                    input_gradients(token, in) += dOut * weights.value(out, in);
                }
            }
        }

        return input_gradients;
    }

    if (output_gradients.shape().size() != 1 || output_gradients.size() != outputSize)
    {
        throw std::invalid_argument("Output gradient must be a 1D tensor matching the layer output size");
    }

    Tensor input_gradients({inputSize});
    for (size_t out = 0; out < outputSize; ++out)
    {
        double dOut = output_gradients(out);
        bias.grad(out) += dOut;

        for (size_t in = 0; in < inputSize; ++in)
        {
            weights.grad(out, in) += dOut * last_input(in);
            input_gradients(in) += dOut * weights.value(out, in);
        }
    }
    return input_gradients;
}

std::vector<Parameter*> LinearLayer::parameters()
{
    return { &weights, &bias };
}
