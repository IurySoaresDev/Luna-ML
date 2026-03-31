#pragma once

#include "tensor.hpp"
#include <vector>
#include <memory>

namespace nf {

// ── GradFunction Base ──
struct GradFunction {
    virtual ~GradFunction() = default;
    virtual std::vector<Tensor> apply(const Tensor& grad_output) = 0;
    std::vector<Tensor> inputs;
    std::vector<Tensor> saved_tensors;
    void save_for_backward(const Tensor& t) { saved_tensors.push_back(t); }
};

// ── NoGradGuard — RAII to disable gradient tracking ──
class NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();
    static bool is_enabled();
private:
    bool prev_;
    static thread_local bool enabled_;
};

// ── Backward Functions ──

struct AddBackward : GradFunction {
    Shape shape_a, shape_b;
    std::vector<Tensor> apply(const Tensor& grad_output) override;
};

struct SubBackward : GradFunction {
    Shape shape_a, shape_b;
    std::vector<Tensor> apply(const Tensor& grad_output) override;
};

struct MulBackward : GradFunction {
    std::vector<Tensor> apply(const Tensor& grad_output) override;
};

struct DivBackward : GradFunction {
    std::vector<Tensor> apply(const Tensor& grad_output) override;
};

struct NegBackward : GradFunction {
    std::vector<Tensor> apply(const Tensor& grad_output) override;
};

struct MatMulBackward : GradFunction {
    std::vector<Tensor> apply(const Tensor& grad_output) override;
};

struct SumBackward : GradFunction {
    Shape input_shape;
    int axis;
    bool keepdim;
    std::vector<Tensor> apply(const Tensor& grad_output) override;
};

struct MeanBackward : GradFunction {
    Shape input_shape;
    int axis;
    bool keepdim;
    std::vector<Tensor> apply(const Tensor& grad_output) override;
};

struct ExpBackward : GradFunction {
    Tensor output;
    std::vector<Tensor> apply(const Tensor& grad_output) override;
};

struct LogBackward : GradFunction {
    std::vector<Tensor> apply(const Tensor& grad_output) override;
};

struct PowBackward : GradFunction {
    float exponent;
    std::vector<Tensor> apply(const Tensor& grad_output) override;
};

struct ReLUBackward : GradFunction {
    std::vector<Tensor> apply(const Tensor& grad_output) override;
};

struct SigmoidBackward : GradFunction {
    Tensor output;
    std::vector<Tensor> apply(const Tensor& grad_output) override;
};

struct TanhBackward : GradFunction {
    Tensor output;
    std::vector<Tensor> apply(const Tensor& grad_output) override;
};

struct SoftmaxBackward : GradFunction {
    Tensor output;
    int axis;
    std::vector<Tensor> apply(const Tensor& grad_output) override;
};

struct ReshapeBackward : GradFunction {
    Shape original_shape;
    std::vector<Tensor> apply(const Tensor& grad_output) override;
};

struct TransposeBackward : GradFunction {
    std::vector<Tensor> apply(const Tensor& grad_output) override;
};

struct FlattenBackward : GradFunction {
    Shape original_shape;
    std::vector<Tensor> apply(const Tensor& grad_output) override;
};

struct ClampBackward : GradFunction {
    float min_val, max_val;
    std::vector<Tensor> apply(const Tensor& grad_output) override;
};

struct RepeatBackward : GradFunction {
    Shape original_shape;
    Shape repeats;
    std::vector<Tensor> apply(const Tensor& grad_output) override;
};

struct ScalarMulBackward : GradFunction {
    float scalar;
    std::vector<Tensor> apply(const Tensor& grad_output) override;
};

struct ScalarAddBackward : GradFunction {
    std::vector<Tensor> apply(const Tensor& grad_output) override;
};

// ── Utility ──
Tensor reduce_grad_for_broadcast(const Tensor& grad, const Shape& target_shape);

} // namespace nf
