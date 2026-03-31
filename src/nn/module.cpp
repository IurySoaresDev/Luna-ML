#include "lunaml/nn/module.hpp"

namespace nf { namespace nn {

std::vector<Tensor*> Module::parameters() {
    std::vector<Tensor*> result;
    for (auto& p : params_) result.push_back(&p);
    return result;
}

void Module::zero_grad() {
    for (auto* p : parameters()) {
        if (p->has_grad()) {
            p->grad().zero_();
        }
    }
}

}} // namespace nf::nn
