#pragma once
#include "lunaml/nn/module.hpp"
#include "lunaml/core/tensor_ops.hpp"

namespace nf { namespace nn {

class ReLU : public Module {
public:
    Tensor forward(const Tensor& input) override { return nf::relu(input); }
    std::string name() const override { return "ReLU"; }
};

class Sigmoid : public Module {
public:
    Tensor forward(const Tensor& input) override { return nf::sigmoid(input); }
    std::string name() const override { return "Sigmoid"; }
};

class Tanh : public Module {
public:
    Tensor forward(const Tensor& input) override { return input.tanh_(); }
    std::string name() const override { return "Tanh"; }
};

class Softmax : public Module {
public:
    Softmax(int axis = -1) : axis_(axis) {}
    Tensor forward(const Tensor& input) override { return nf::softmax(input, axis_); }
    std::string name() const override { return "Softmax"; }
private:
    int axis_;
};

class LeakyReLU : public Module {
public:
    LeakyReLU(float negative_slope = 0.01f) : slope_(negative_slope) {}
    Tensor forward(const Tensor& input) override;
    std::string name() const override { return "LeakyReLU"; }
private:
    float slope_;
};

}} // namespace nf::nn
