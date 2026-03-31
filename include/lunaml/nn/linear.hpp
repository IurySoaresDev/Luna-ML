#pragma once
#include "lunaml/nn/module.hpp"

namespace nf { namespace nn {

class Linear : public Module {
public:
    Linear(size_t in_features, size_t out_features, bool bias = true);
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor*> parameters() override;
    std::string name() const override { return "Linear(" + std::to_string(in_features_) + ", " + std::to_string(out_features_) + ")"; }

    Tensor& weight() { return weight_; }
    Tensor& bias() { return bias_; }

private:
    size_t in_features_, out_features_;
    bool use_bias_;
    Tensor weight_;
    Tensor bias_;
};

}} // namespace nf::nn
