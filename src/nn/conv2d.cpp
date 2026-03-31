#include "lunaml/nn/conv2d.hpp"
#include "lunaml/core/random.hpp"
#include <cmath>

namespace nf { namespace nn {

Conv2d::Conv2d(size_t in_ch, size_t out_ch, size_t ks, size_t stride, size_t padding)
    : in_channels_(in_ch), out_channels_(out_ch), kernel_size_(ks),
      stride_(stride), padding_(padding) {
    float k = 1.0f / static_cast<float>(in_ch * ks * ks);
    float bound = std::sqrt(k);
    weight_ = Tensor({out_ch, in_ch, ks, ks}, true);
    weight_.uniform_(-bound, bound);
    bias_ = Tensor({out_ch}, true);
    bias_.uniform_(-bound, bound);
}

Tensor Conv2d::forward(const Tensor& input) {
    // input: [batch, in_channels, H, W]
    size_t batch = input.shape(0);
    size_t H = input.shape(2), W = input.shape(3);
    size_t H_out = (H + 2 * padding_ - kernel_size_) / stride_ + 1;
    size_t W_out = (W + 2 * padding_ - kernel_size_) / stride_ + 1;

    Tensor output({batch, out_channels_, H_out, W_out},
                  input.requires_grad() || weight_.requires_grad());

    for (size_t b = 0; b < batch; ++b) {
        for (size_t oc = 0; oc < out_channels_; ++oc) {
            for (size_t oh = 0; oh < H_out; ++oh) {
                for (size_t ow = 0; ow < W_out; ++ow) {
                    float sum = bias_.data()[oc];
                    for (size_t ic = 0; ic < in_channels_; ++ic) {
                        for (size_t kh = 0; kh < kernel_size_; ++kh) {
                            for (size_t kw = 0; kw < kernel_size_; ++kw) {
                                int ih = static_cast<int>(oh * stride_ + kh) - static_cast<int>(padding_);
                                int iw = static_cast<int>(ow * stride_ + kw) - static_cast<int>(padding_);
                                if (ih >= 0 && ih < static_cast<int>(H) &&
                                    iw >= 0 && iw < static_cast<int>(W)) {
                                    size_t in_idx = ((b * in_channels_ + ic) * H + ih) * W + iw;
                                    size_t w_idx = ((oc * in_channels_ + ic) * kernel_size_ + kh) * kernel_size_ + kw;
                                    sum += input.data()[in_idx] * weight_.data()[w_idx];
                                }
                            }
                        }
                    }
                    size_t out_idx = ((b * out_channels_ + oc) * H_out + oh) * W_out + ow;
                    output.data()[out_idx] = sum;
                }
            }
        }
    }
    return output;
}

std::vector<Tensor*> Conv2d::parameters() {
    return {&weight_, &bias_};
}

std::string Conv2d::name() const {
    return "Conv2d(" + std::to_string(in_channels_) + ", " +
           std::to_string(out_channels_) + ", kernel=" + std::to_string(kernel_size_) + ")";
}

}} // namespace nf::nn
