#include "lunaml/optim/adam.hpp"
#include "lunaml/core/autograd.hpp"
#include <cmath>

namespace nf { namespace optim {

Adam::Adam(std::vector<Tensor*> params, float lr, float beta1, float beta2, float eps)
    : Optimizer(params, lr), beta1_(beta1), beta2_(beta2), eps_(eps) {
    for (auto* p : params_) {
        m_.push_back(Tensor::zeros(p->shape()));
        v_.push_back(Tensor::zeros(p->shape()));
    }
}

void Adam::step() {
    NoGradGuard guard;
    t_++;
    float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(t_));
    float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(t_));

    for (size_t i = 0; i < params_.size(); ++i) {
        auto* p = params_[i];
        if (!p->has_grad()) continue;
        auto& grad = p->grad();

        for (size_t j = 0; j < p->size(); ++j) {
            float g = grad.data()[j];
            m_[i].data()[j] = beta1_ * m_[i].data()[j] + (1.0f - beta1_) * g;
            v_[i].data()[j] = beta2_ * v_[i].data()[j] + (1.0f - beta2_) * g * g;

            float m_hat = m_[i].data()[j] / bc1;
            float v_hat = v_[i].data()[j] / bc2;

            p->data()[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }
}

}} // namespace nf::optim
