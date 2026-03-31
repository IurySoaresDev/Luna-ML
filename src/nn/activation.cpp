#include "lunaml/nn/activation.hpp"

namespace nf { namespace nn {

Tensor LeakyReLU::forward(const Tensor& input) {
    Tensor result(input.shape(), input.requires_grad());
    for (size_t i = 0; i < input.size(); ++i) {
        result.data()[i] = input.data()[i] > 0 ? input.data()[i] : slope_ * input.data()[i];
    }
    // Autograd: treat as clamp-like
    if (input.requires_grad() && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<ReLUBackward>();
        fn->inputs = {input};
        // Use a custom approach: store input and slope in saved_tensors
        fn->save_for_backward(input);
        result.set_grad_fn(fn);
    }
    return result;
}

}} // namespace nf::nn
