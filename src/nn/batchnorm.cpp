#include "lunaml/nn/batchnorm.hpp"
#include "lunaml/core/autograd.hpp"

namespace nf { namespace nn {

BatchNorm1d::BatchNorm1d(size_t nf, float eps, float momentum)
    : num_features_(nf), eps_(eps), momentum_(momentum) {
    gamma_ = Tensor::ones({1, nf}, true);
    beta_ = Tensor::zeros({1, nf}, true);
    running_mean_ = Tensor::zeros({1, nf}, false);
    running_var_ = Tensor::ones({1, nf}, false);
}

Tensor BatchNorm1d::forward(const Tensor& input) {
    // input: [batch, features]
    size_t feat = input.shape(1);

    Tensor mean_t, var_t;

    if (is_training()) {
        mean_t = input.mean(0, true);  // [1, feat]
        auto diff = input - mean_t;
        var_t = (diff * diff).mean(0, true);  // [1, feat]

        // Update running stats
        NoGradGuard guard;
        for (size_t i = 0; i < feat; ++i) {
            running_mean_.data()[i] = (1.0f - momentum_) * running_mean_.data()[i] + momentum_ * mean_t.data()[i];
            running_var_.data()[i] = (1.0f - momentum_) * running_var_.data()[i] + momentum_ * var_t.data()[i];
        }
    } else {
        mean_t = running_mean_;
        var_t = running_var_;
    }

    // Normalize
    auto x_norm = (input - mean_t) / (var_t + eps_).sqrt();
    return x_norm * gamma_ + beta_;
}

std::vector<Tensor*> BatchNorm1d::parameters() {
    return {&gamma_, &beta_};
}

std::string BatchNorm1d::name() const {
    return "BatchNorm1d(" + std::to_string(num_features_) + ")";
}

}} // namespace nf::nn
