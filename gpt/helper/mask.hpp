#ifndef MASK_HPP
#define MASK_HPP

#include "../Tensor/Tensor.hpp"
#include <limits>
#include <stdexcept>
#include <vector>

class Mask
{
    public:
        static void causalMask(Tensor &scores)
        {
            const std::vector<size_t> &shape = scores.shape();
            const size_t ndim = shape.size();
            if (ndim < 2)
            {
                throw std::invalid_argument("causalMask(): scores must have at least 2 dimensions");
            }

            const size_t seqLenQ = shape[ndim - 2];
            const size_t seqLenK = shape[ndim - 1];
            const size_t batch = scores.size() / (seqLenQ * seqLenK);

            for (size_t b = 0; b < batch; b++)
            {
                const size_t base = b * seqLenQ * seqLenK;
                for (size_t row = 0; row < seqLenQ; row++)
                {
                    for (size_t col = row + 1; col < seqLenK; col++)
                    {
                        scores(base + row * seqLenK + col) = -std::numeric_limits<double>::infinity();
                    }
                }
            }
        }
};

#endif