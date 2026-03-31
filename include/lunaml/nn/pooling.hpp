#pragma once
#include "lunaml/nn/module.hpp"

namespace nf { namespace nn {

class MaxPool2d : public Module {
public:
    MaxPool2d(size_t kernel_size, size_t stride = 0);
    Tensor forward(const Tensor& input) override;
    std::string name() const override;
private:
    size_t kernel_size_, stride_;
};

class AvgPool2d : public Module {
public:
    AvgPool2d(size_t kernel_size, size_t stride = 0);
    Tensor forward(const Tensor& input) override;
    std::string name() const override;
private:
    size_t kernel_size_, stride_;
};

}} // namespace nf::nn
