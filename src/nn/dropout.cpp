#include "lunaml/nn/dropout.hpp"
#include "lunaml/core/random.hpp"

namespace nf { namespace nn {

Tensor Dropout::forward(const Tensor& input) {
    if (!is_training() || p_ == 0.0f) return input;
    Tensor mask(input.shape(), false);
    auto& gen = Random::generator();
    std::bernoulli_distribution dist(1.0 - static_cast<double>(p_));
    float scale = 1.0f / (1.0f - p_);
    for (size_t i = 0; i < input.size(); ++i) {
        mask.data()[i] = dist(gen) ? scale : 0.0f;
    }
    return input * mask;
}

Tensor Flatten::forward(const Tensor& input) {
    return input.flatten(start_dim_);
}

}} // namespace nf::nn
