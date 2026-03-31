#include "lunaml/optim/rmsprop.hpp"
#include "lunaml/core/autograd.hpp"
#include <cmath>

namespace nf { namespace optim {

RMSProp::RMSProp(std::vector<Tensor*> params, float lr, float alpha, float eps)
    : Optimizer(params, lr), alpha_(alpha), eps_(eps) {
    for (auto* p : params_) {
        sq_avg_.push_back(Tensor::zeros(p->shape()));
    }
}

void RMSProp::step() {
    NoGradGuard guard;
    for (size_t i = 0; i < params_.size(); ++i) {
        auto* p = params_[i];
        if (!p->has_grad()) continue;
        auto& grad = p->grad();
        for (size_t j = 0; j < p->size(); ++j) {
            float g = grad.data()[j];
            sq_avg_[i].data()[j] = alpha_ * sq_avg_[i].data()[j] + (1.0f - alpha_) * g * g;
            p->data()[j] -= lr_ * g / (std::sqrt(sq_avg_[i].data()[j]) + eps_);
        }
    }
}

}} // namespace nf::optim
