#include "ActivationFunction.hpp"

#include <cmath>
#include <algorithm>

namespace Activation
{

    Tensor softmaxForward(const Tensor &input)
    {
        Tensor output(input.shape());
        const auto &shape = input.shape();
        const size_t lastDim = shape.empty() ? input.size() : shape.back();

        if (lastDim == 0)
            return output;

        const size_t rows = input.size() / lastDim;

        for (size_t r = 0; r < rows; r++)
        {
            const size_t base = r * lastDim;

            double maxValue = input(base);
            for (size_t c = 1; c < lastDim; c++)
                maxValue = std::max(maxValue, input(base + c));

            double sum = 0.0;
            for (size_t c = 0; c < lastDim; c++)
            {
                double e = std::exp(input(base + c) - maxValue);
                output(base + c) = e;
                sum += e;
            }

            for (size_t c = 0; c < lastDim; c++)
                output(base + c) /= sum;
        }

        return output;
    }

    Tensor softmaxBackward(const Tensor &gradOutput, const Tensor &softmaxOutput)
    {
        Tensor gradInput(gradOutput.shape());
        const auto &shape = gradOutput.shape();
        const size_t lastDim = shape.empty() ? gradOutput.size() : shape.back();

        if (lastDim == 0)
            return gradInput;

        const size_t rows = gradOutput.size() / lastDim;

        for (size_t r = 0; r < rows; r++)
        {
            const size_t base = r * lastDim;

            double dot = 0.0;
            for (size_t c = 0; c < lastDim; c++)
                dot += gradOutput(base + c) * softmaxOutput(base + c);

            for (size_t c = 0; c < lastDim; c++)
                gradInput(base + c) = softmaxOutput(base + c) * (gradOutput(base + c) - dot);
        }

        return gradInput;
    }

    Tensor geluForward(const Tensor &input)
    {
        Tensor output(input.shape());
        constexpr double sqrt_2_over_pi = 0.7978845608028654;

        for (size_t i = 0; i < input.size(); i++)
        {
            double x = input(i);
            double cubic = x * x * x;
            double inner = sqrt_2_over_pi * (x + 0.044715 * cubic);

            output(i) = 0.5 * x * (1.0 + std::tanh(inner));
        }

        return output;
    }

    Tensor geluBackward(const Tensor &gradOutput, const Tensor &input)
    {
        Tensor gradInput(input.shape());
        constexpr double sqrt_2_over_pi = 0.7978845608028654;

        for (size_t i = 0; i < input.size(); i++)
        {
            double x = input(i);
            double x2 = x * x;
            double cubic = x * x2;
            double inner = sqrt_2_over_pi * (x + 0.044715 * cubic);
            double tanhValue = std::tanh(inner);
            double sech2 = 1.0 - tanhValue * tanhValue;
            double derivative = 0.5 * (1.0 + tanhValue) + 0.5 * x * sech2 * sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x2);

            gradInput(i) = gradOutput(i) * derivative;
        }
        return gradInput;
    }
}