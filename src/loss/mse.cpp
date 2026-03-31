#include "lunaml/loss/mse.hpp"

namespace nf { namespace loss {

Tensor MSE::forward(const Tensor& prediction, const Tensor& target) {
    auto diff = prediction - target;
    auto sq = diff * diff;
    return sq.mean();
}

}} // namespace nf::loss
