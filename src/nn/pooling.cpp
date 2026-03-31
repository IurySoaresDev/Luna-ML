#include "lunaml/nn/pooling.hpp"

namespace nf { namespace nn {

MaxPool2d::MaxPool2d(size_t ks, size_t stride) : kernel_size_(ks), stride_(stride == 0 ? ks : stride) {}

Tensor MaxPool2d::forward(const Tensor& input) {
    size_t batch = input.shape(0), channels = input.shape(1);
    size_t H = input.shape(2), W = input.shape(3);
    size_t H_out = (H - kernel_size_) / stride_ + 1;
    size_t W_out = (W - kernel_size_) / stride_ + 1;
    Tensor output({batch, channels, H_out, W_out}, false);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t oh = 0; oh < H_out; ++oh) {
                for (size_t ow = 0; ow < W_out; ++ow) {
                    float max_val = -1e30f;
                    for (size_t kh = 0; kh < kernel_size_; ++kh) {
                        for (size_t kw = 0; kw < kernel_size_; ++kw) {
                            size_t ih = oh * stride_ + kh;
                            size_t iw = ow * stride_ + kw;
                            size_t idx = ((b * channels + c) * H + ih) * W + iw;
                            max_val = std::max(max_val, input.data()[idx]);
                        }
                    }
                    output.data()[((b * channels + c) * H_out + oh) * W_out + ow] = max_val;
                }
            }
        }
    }
    return output;
}

std::string MaxPool2d::name() const { return "MaxPool2d(" + std::to_string(kernel_size_) + ")"; }

AvgPool2d::AvgPool2d(size_t ks, size_t stride) : kernel_size_(ks), stride_(stride == 0 ? ks : stride) {}

Tensor AvgPool2d::forward(const Tensor& input) {
    size_t batch = input.shape(0), channels = input.shape(1);
    size_t H = input.shape(2), W = input.shape(3);
    size_t H_out = (H - kernel_size_) / stride_ + 1;
    size_t W_out = (W - kernel_size_) / stride_ + 1;
    Tensor output({batch, channels, H_out, W_out}, false);
    float area = static_cast<float>(kernel_size_ * kernel_size_);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t oh = 0; oh < H_out; ++oh) {
                for (size_t ow = 0; ow < W_out; ++ow) {
                    float sum = 0;
                    for (size_t kh = 0; kh < kernel_size_; ++kh) {
                        for (size_t kw = 0; kw < kernel_size_; ++kw) {
                            size_t ih = oh * stride_ + kh;
                            size_t iw = ow * stride_ + kw;
                            sum += input.data()[((b * channels + c) * H + ih) * W + iw];
                        }
                    }
                    output.data()[((b * channels + c) * H_out + oh) * W_out + ow] = sum / area;
                }
            }
        }
    }
    return output;
}

std::string AvgPool2d::name() const { return "AvgPool2d(" + std::to_string(kernel_size_) + ")"; }

}} // namespace nf::nn
