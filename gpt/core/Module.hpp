#pragma once
#include "Parameter.hpp"
#include <vector>

class Module
{
public:
    virtual ~Module() = default;
    virtual std::vector<Parameter*> parameters() = 0;

    virtual void train() { training_ = true; }
    virtual void eval() { training_ = false; }
    bool is_training() const { return training_; }

    virtual void zero_grad()
    {
        for (Parameter* p : parameters()) {
            if (p) p->zero_grad();
        }
    }

protected:
    bool training_ = true;
};
