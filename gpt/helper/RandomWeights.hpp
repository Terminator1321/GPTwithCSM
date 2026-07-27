#ifndef RANDOMWEIGHTS_HPP
#define RANDOMWEIGHTS_HPP
#include <random>

inline double get_weight(double min_value = -0.05, double max_value = 0.05)
{
    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(min_value, max_value);
    return dist(gen);
}
#endif