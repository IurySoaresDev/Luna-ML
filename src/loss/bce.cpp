#include "lunaml/loss/bce.hpp"

namespace nf { namespace loss {

Tensor BCE::forward(const Tensor& prediction, const Tensor& target) {
    // BCE = -mean(target * log(pred) + (1-target) * log(1-pred))
    auto pred_clamped = prediction.clamp(1e-7f, 1.0f - 1e-7f);
    auto term1 = target * pred_clamped.log();
    auto term2 = (Tensor::ones(target.shape()) - target) *
                 (Tensor::ones(pred_clamped.shape()) - pred_clamped).log();
    auto loss = -(term1 + term2).mean();
    return loss;
}

}} // namespace nf::loss
