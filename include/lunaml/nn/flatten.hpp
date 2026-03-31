#pragma once
#include "lunaml/nn/module.hpp"

namespace nf { namespace nn {

// Flatten as standalone header inclusion
class FlattenLayer : public Module {
public:
    FlattenLayer(size_t start_dim = 1) : start_dim_(start_dim) {}
    Tensor forward(const Tensor& input) override { return input.flatten(start_dim_); }
    std::string name() const override { return "Flatten"; }
private:
    size_t start_dim_;
};

}} // namespace nf::nn
