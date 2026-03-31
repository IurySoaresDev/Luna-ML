#include "lunaml/optim/sgd.hpp"
#include "lunaml/core/autograd.hpp"

namespace nf { namespace optim {

SGD::SGD(std::vector<Tensor*> params, float lr, float momentum, float weight_decay)
    : Optimizer(params, lr), momentum_(momentum), weight_decay_(weight_decay) {
    if (momentum_ > 0) {
        for (auto* p : params_) {
            velocity_.push_back(Tensor::zeros(p->shape()));
        }
    }
}

void SGD::step() {
    NoGradGuard guard;
    for (size_t i = 0; i < params_.size(); ++i) {
        auto* p = params_[i];
        if (!p->has_grad()) continue;
        auto& grad = p->grad();

        if (weight_decay_ > 0) {
            for (size_t j = 0; j < p->size(); ++j) {
                grad.data()[j] += weight_decay_ * p->data()[j];
            }
        }

        if (momentum_ > 0) {
            for (size_t j = 0; j < p->size(); ++j) {
                velocity_[i].data()[j] = momentum_ * velocity_[i].data()[j] + grad.data()[j];
                p->data()[j] -= lr_ * velocity_[i].data()[j];
            }
        } else {
            for (size_t j = 0; j < p->size(); ++j) {
                p->data()[j] -= lr_ * grad.data()[j];
            }
        }
    }
}

}} // namespace nf::optim
