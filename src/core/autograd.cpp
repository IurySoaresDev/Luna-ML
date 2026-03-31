#include "lunaml/core/autograd.hpp"

namespace nf {

// ── NoGradGuard ──
thread_local bool NoGradGuard::enabled_ = false;

NoGradGuard::NoGradGuard() : prev_(enabled_) { enabled_ = true; }
NoGradGuard::~NoGradGuard() { enabled_ = prev_; }
bool NoGradGuard::is_enabled() { return enabled_; }

// ── Broadcast gradient reduction ──
Tensor reduce_grad_for_broadcast(const Tensor& grad, const Shape& target_shape) {
    Shape grad_shape = grad.shape();
    if (grad_shape == target_shape) return grad;

    Tensor result = grad;

    // Sum over leading dims added by broadcasting
    while (result.ndim() > target_shape.size()) {
        result = result.sum(0, false);
    }

    // Sum over dims that were broadcast from size 1
    for (size_t i = 0; i < target_shape.size(); ++i) {
        if (target_shape[i] == 1 && result.shape(i) != 1) {
            result = result.sum(static_cast<int>(i), true);
        }
    }
    return result;
}

// ── AddBackward ──
std::vector<Tensor> AddBackward::apply(const Tensor& grad_output) {
    auto ga = reduce_grad_for_broadcast(grad_output, shape_a);
    auto gb = reduce_grad_for_broadcast(grad_output, shape_b);
    return {ga, gb};
}

// ── SubBackward ──
std::vector<Tensor> SubBackward::apply(const Tensor& grad_output) {
    auto ga = reduce_grad_for_broadcast(grad_output, shape_a);
    auto neg_grad = -grad_output;
    auto gb = reduce_grad_for_broadcast(neg_grad, shape_b);
    return {ga, gb};
}

// ── MulBackward ──
std::vector<Tensor> MulBackward::apply(const Tensor& grad_output) {
    auto& a = saved_tensors[0];
    auto& b = saved_tensors[1];
    auto ga = reduce_grad_for_broadcast(grad_output * b, a.shape());
    auto gb = reduce_grad_for_broadcast(grad_output * a, b.shape());
    return {ga, gb};
}

// ── DivBackward ──
std::vector<Tensor> DivBackward::apply(const Tensor& grad_output) {
    auto& a = saved_tensors[0];
    auto& b = saved_tensors[1];
    auto ga = reduce_grad_for_broadcast(grad_output / b, a.shape());
    auto gb = reduce_grad_for_broadcast(-(grad_output * a) / (b * b), b.shape());
    return {ga, gb};
}

// ── NegBackward ──
std::vector<Tensor> NegBackward::apply(const Tensor& grad_output) {
    return {-grad_output};
}

// ── MatMulBackward ──
std::vector<Tensor> MatMulBackward::apply(const Tensor& grad_output) {
    auto& a = saved_tensors[0];
    auto& b = saved_tensors[1];
    // grad_a = grad_output @ b^T
    // grad_b = a^T @ grad_output
    auto ga = grad_output.matmul(b.transpose());
    auto gb = a.transpose().matmul(grad_output);
    return {ga, gb};
}

// ── SumBackward ──
std::vector<Tensor> SumBackward::apply(const Tensor& grad_output) {
    if (axis == -1) {
        // Global sum: broadcast scalar grad to input shape
        Tensor result = Tensor::ones(input_shape);
        float g = grad_output.data()[0];
        for (size_t i = 0; i < result.size(); ++i) result.data()[i] = g;
        return {result};
    }
    // Axis sum: expand grad along the summed axis
    Tensor result = Tensor::zeros(input_shape);
    size_t a = static_cast<size_t>(axis);
    size_t outer = 1, inner = 1;
    for (size_t i = 0; i < a; ++i) outer *= input_shape[i];
    for (size_t i = a + 1; i < input_shape.size(); ++i) inner *= input_shape[i];
    size_t dim_size = input_shape[a];
    for (size_t o = 0; o < outer; ++o) {
        for (size_t in = 0; in < inner; ++in) {
            float g = grad_output.data()[o * inner + in];
            for (size_t d = 0; d < dim_size; ++d) {
                result.data()[(o * dim_size + d) * inner + in] = g;
            }
        }
    }
    return {result};
}

// ── MeanBackward ──
std::vector<Tensor> MeanBackward::apply(const Tensor& grad_output) {
    float n;
    if (axis == -1) {
        n = static_cast<float>(shape_size(input_shape));
    } else {
        n = static_cast<float>(input_shape[static_cast<size_t>(axis)]);
    }

    // Reuse SumBackward logic then divide by n
    SumBackward sb;
    sb.input_shape = input_shape;
    sb.axis = axis;
    sb.keepdim = keepdim;
    auto grads = sb.apply(grad_output);
    for (size_t i = 0; i < grads[0].size(); ++i) {
        grads[0].data()[i] /= n;
    }
    return grads;
}

// ── ExpBackward ──
std::vector<Tensor> ExpBackward::apply(const Tensor& grad_output) {
    // d/dx exp(x) = exp(x) = output
    return {grad_output * output};
}

// ── LogBackward ──
std::vector<Tensor> LogBackward::apply(const Tensor& grad_output) {
    auto& input = saved_tensors[0];
    Tensor inv(input.shape(), false);
    for (size_t i = 0; i < input.size(); ++i) {
        inv.data()[i] = 1.0f / (input.data()[i] + 1e-8f);
    }
    return {grad_output * inv};
}

// ── PowBackward ──
std::vector<Tensor> PowBackward::apply(const Tensor& grad_output) {
    auto& input = saved_tensors[0];
    Tensor grad_input(input.shape(), false);
    for (size_t i = 0; i < input.size(); ++i) {
        grad_input.data()[i] = exponent * std::pow(input.data()[i], exponent - 1.0f);
    }
    return {grad_output * grad_input};
}

// ── ReLUBackward ──
std::vector<Tensor> ReLUBackward::apply(const Tensor& grad_output) {
    auto& input = saved_tensors[0];
    Tensor result(grad_output.shape(), false);
    for (size_t i = 0; i < result.size(); ++i) {
        result.data()[i] = input.data()[i] > 0.0f ? grad_output.data()[i] : 0.0f;
    }
    return {result};
}

// ── SigmoidBackward ──
std::vector<Tensor> SigmoidBackward::apply(const Tensor& grad_output) {
    // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    Tensor result(output.shape(), false);
    for (size_t i = 0; i < result.size(); ++i) {
        float s = output.data()[i];
        result.data()[i] = grad_output.data()[i] * s * (1.0f - s);
    }
    return {result};
}

// ── TanhBackward ──
std::vector<Tensor> TanhBackward::apply(const Tensor& grad_output) {
    Tensor result(output.shape(), false);
    for (size_t i = 0; i < result.size(); ++i) {
        float t = output.data()[i];
        result.data()[i] = grad_output.data()[i] * (1.0f - t * t);
    }
    return {result};
}

// ── SoftmaxBackward ──
std::vector<Tensor> SoftmaxBackward::apply(const Tensor& grad_output) {
    // Simplified: softmax grad for last axis
    Tensor result(output.shape(), false);
    if (output.ndim() == 2) {
        size_t batch = output.shape(0), classes = output.shape(1);
        for (size_t b = 0; b < batch; ++b) {
            // s_i * (delta_ij - s_j) * dL/ds_j summed over j
            for (size_t i = 0; i < classes; ++i) {
                float sum = 0.0f;
                float si = output.data()[b * classes + i];
                for (size_t j = 0; j < classes; ++j) {
                    float sj = output.data()[b * classes + j];
                    float dij = (i == j) ? 1.0f : 0.0f;
                    sum += (dij - sj) * grad_output.data()[b * classes + j];
                }
                result.data()[b * classes + i] = si * sum;
            }
        }
    } else {
        // 1D case
        size_t n = output.size();
        for (size_t i = 0; i < n; ++i) {
            float sum = 0.0f;
            float si = output.data()[i];
            for (size_t j = 0; j < n; ++j) {
                float sj = output.data()[j];
                float dij = (i == j) ? 1.0f : 0.0f;
                sum += (dij - sj) * grad_output.data()[j];
            }
            result.data()[i] = si * sum;
        }
    }
    return {result};
}

// ── ReshapeBackward ──
std::vector<Tensor> ReshapeBackward::apply(const Tensor& grad_output) {
    return {grad_output.reshape(original_shape)};
}

// ── TransposeBackward ──
std::vector<Tensor> TransposeBackward::apply(const Tensor& grad_output) {
    return {grad_output.transpose()};
}

// ── FlattenBackward ──
std::vector<Tensor> FlattenBackward::apply(const Tensor& grad_output) {
    return {grad_output.reshape(original_shape)};
}

// ── ClampBackward ──
std::vector<Tensor> ClampBackward::apply(const Tensor& grad_output) {
    auto& input = saved_tensors[0];
    Tensor result(input.shape(), false);
    for (size_t i = 0; i < input.size(); ++i) {
        float v = input.data()[i];
        result.data()[i] = (v > min_val && v < max_val) ? grad_output.data()[i] : 0.0f;
    }
    return {result};
}

// ── RepeatBackward ──
std::vector<Tensor> RepeatBackward::apply(const Tensor& grad_output) {
    Tensor result = Tensor::zeros(original_shape);
    size_t total = grad_output.size();
    auto out_shape = grad_output.shape();
    for (size_t i = 0; i < total; ++i) {
        auto idx = unravel_index(i, out_shape);
        size_t src_flat = 0;
        auto src_strides = compute_strides(original_shape);
        for (size_t d = 0; d < original_shape.size(); ++d) {
            src_flat += (idx[d] % original_shape[d]) * src_strides[d];
        }
        result.data()[src_flat] += grad_output.data()[i];
    }
    return {result};
}

// ── ScalarMulBackward ──
std::vector<Tensor> ScalarMulBackward::apply(const Tensor& grad_output) {
    return {grad_output * scalar};
}

// ── ScalarAddBackward ──
std::vector<Tensor> ScalarAddBackward::apply(const Tensor& grad_output) {
    return {grad_output};
}

} // namespace nf
