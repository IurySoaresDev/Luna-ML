#pragma once
#include "lunaml/optim/optimizer.hpp"

namespace nf { namespace optim {

class RMSProp : public Optimizer {
public:
    RMSProp(std::vector<Tensor*> params, float lr = 0.01f,
            float alpha = 0.99f, float eps = 1e-8f);
    void step() override;

private:
    float alpha_, eps_;
    std::vector<Tensor> sq_avg_;
};

}} // namespace nf::optim
