#pragma once
#include "lunaml/optim/optimizer.hpp"

namespace nf { namespace optim {

class Adam : public Optimizer {
public:
    Adam(std::vector<Tensor*> params, float lr = 0.001f,
         float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);
    void step() override;

private:
    float beta1_, beta2_, eps_;
    size_t t_ = 0;
    std::vector<Tensor> m_; // First moment
    std::vector<Tensor> v_; // Second moment
};

}} // namespace nf::optim
