#pragma once
#include "lunaml/nn/module.hpp"

namespace nf { namespace nn {

class Conv2d : public Module {
public:
    Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size,
           size_t stride = 1, size_t padding = 0);
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor*> parameters() override;
    std::string name() const override;

private:
    size_t in_channels_, out_channels_, kernel_size_, stride_, padding_;
    Tensor weight_; // [out_channels, in_channels, kernel_size, kernel_size]
    Tensor bias_;   // [out_channels]
};

}} // namespace nf::nn
