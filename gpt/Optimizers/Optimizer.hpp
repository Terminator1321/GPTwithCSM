#pragma once
#include "../core/Parameter.hpp"
#include <vector>

class Optimizer
{
public:
    virtual ~Optimizer() = default;
    virtual void step(Parameter& param) = 0;

    void step(const std::vector<Parameter*>& params)
    {
        for (Parameter* p : params)
        {
            if (p) step(*p);
        }
        ++timestep_;
    }

    size_t timestep() const { return timestep_; }

protected:
    size_t timestep_ = 0;
};
