#pragma once
#include "lunaml/core/tensor.hpp"
#include <vector>

namespace nf { namespace optim {

class Optimizer {
public:
    Optimizer(std::vector<Tensor*> params, float lr) : params_(params), lr_(lr) {}
    virtual ~Optimizer() = default;
    virtual void step() = 0;

    void zero_grad() {
        for (auto* p : params_) {
            if (p->has_grad()) p->grad().zero_();
        }
    }

    float learning_rate() const { return lr_; }
    void set_learning_rate(float lr) { lr_ = lr; }

protected:
    std::vector<Tensor*> params_;
    float lr_;
};

}} // namespace nf::optim
