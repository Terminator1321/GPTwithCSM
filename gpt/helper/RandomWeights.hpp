#ifndef RANDOMWEIGHTS_HPP
#define RANDOMWEIGHTS_HPP
#include <random>

inline double get_weight(double min_value = -0.05, double max_value = 0.05) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<double> distrib(min_value, max_value);
    return distrib(gen);
}
#endif