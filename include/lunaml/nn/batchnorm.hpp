#pragma once
#include "lunaml/nn/module.hpp"

namespace nf { namespace nn {

class BatchNorm1d : public Module {
public:
    BatchNorm1d(size_t num_features, float eps = 1e-5f, float momentum = 0.1f);
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor*> parameters() override;
    std::string name() const override;

private:
    size_t num_features_;
    float eps_, momentum_;
    Tensor gamma_, beta_;
    Tensor running_mean_, running_var_;
};

}} // namespace nf::nn
