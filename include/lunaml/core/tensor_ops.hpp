#pragma once
#include "lunaml/core/tensor.hpp"
#include "lunaml/core/autograd.hpp"

namespace nf {

// Free function tensor operations
inline Tensor relu(const Tensor& input) {
    Tensor result(input.shape(), input.requires_grad());
    for (size_t i = 0; i < input.size(); ++i)
        result.data()[i] = std::max(0.0f, input.data()[i]);
    if (input.requires_grad() && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<ReLUBackward>();
        fn->inputs = {input};
        fn->save_for_backward(input);
        result.set_grad_fn(fn);
    }
    return result;
}

inline Tensor sigmoid(const Tensor& input) {
    Tensor result(input.shape(), input.requires_grad());
    for (size_t i = 0; i < input.size(); ++i)
        result.data()[i] = 1.0f / (1.0f + std::exp(-input.data()[i]));
    if (input.requires_grad() && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<SigmoidBackward>();
        fn->output = result;
        fn->inputs = {input};
        result.set_grad_fn(fn);
    }
    return result;
}

inline Tensor softmax(const Tensor& input, int axis = -1) {
    Tensor result(input.shape(), input.requires_grad());
    if (input.ndim() == 2 && (axis == 1 || axis == -1)) {
        size_t batch = input.shape(0), classes = input.shape(1);
        for (size_t b = 0; b < batch; ++b) {
            float max_val = input.data()[b * classes];
            for (size_t c = 1; c < classes; ++c)
                max_val = std::max(max_val, input.data()[b * classes + c]);
            float sum = 0;
            for (size_t c = 0; c < classes; ++c) {
                result.data()[b * classes + c] = std::exp(input.data()[b * classes + c] - max_val);
                sum += result.data()[b * classes + c];
            }
            for (size_t c = 0; c < classes; ++c)
                result.data()[b * classes + c] /= sum;
        }
    } else {
        float max_val = input.data()[0];
        for (size_t i = 1; i < input.size(); ++i)
            max_val = std::max(max_val, input.data()[i]);
        float sum = 0;
        for (size_t i = 0; i < input.size(); ++i) {
            result.data()[i] = std::exp(input.data()[i] - max_val);
            sum += result.data()[i];
        }
        for (size_t i = 0; i < input.size(); ++i)
            result.data()[i] /= sum;
    }
    if (input.requires_grad() && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<SoftmaxBackward>();
        fn->output = result;
        fn->axis = axis;
        fn->inputs = {input};
        result.set_grad_fn(fn);
    }
    return result;
}

inline Tensor log_softmax(const Tensor& input, int axis = -1) {
    return softmax(input, axis).log();
}

} // namespace nf
