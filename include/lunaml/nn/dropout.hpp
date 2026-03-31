#pragma once
#include "lunaml/nn/module.hpp"

namespace nf { namespace nn {

class Dropout : public Module {
public:
    Dropout(float p = 0.5f) : p_(p) {}
    Tensor forward(const Tensor& input) override;
    std::string name() const override { return "Dropout(" + std::to_string(p_) + ")"; }
private:
    float p_;
};

class Flatten : public Module {
public:
    Flatten(size_t start_dim = 1) : start_dim_(start_dim) {}
    Tensor forward(const Tensor& input) override;
    std::string name() const override { return "Flatten"; }
private:
    size_t start_dim_;
};

}} // namespace nf::nn
