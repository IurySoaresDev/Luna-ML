#include "lunaml/nn/linear.hpp"
#include "lunaml/core/random.hpp"
#include <cmath>

namespace nf { namespace nn {

Linear::Linear(size_t in_features, size_t out_features, bool bias)
    : in_features_(in_features), out_features_(out_features), use_bias_(bias) {
    // Kaiming uniform initialization
    float k = 1.0f / static_cast<float>(in_features);
    float bound = std::sqrt(k);

    weight_ = Tensor({out_features, in_features}, true);
    weight_.uniform_(-bound, bound);

    if (use_bias_) {
        bias_ = Tensor({1, out_features}, true);
        bias_.uniform_(-bound, bound);
    }
}

Tensor Linear::forward(const Tensor& input) {
    // input: [batch, in_features]
    // weight: [out_features, in_features]
    // output: [batch, out_features]
    Tensor output = input.matmul(weight_.transpose());
    if (use_bias_) {
        output = output + bias_;
    }
    return output;
}

std::vector<Tensor*> Linear::parameters() {
    std::vector<Tensor*> params = {&weight_};
    if (use_bias_) params.push_back(&bias_);
    return params;
}

}} // namespace nf::nn
