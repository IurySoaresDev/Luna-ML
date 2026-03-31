#pragma once

#include "types.hpp"
#include <memory>

namespace nf {

// Forward declarations
struct GradFunction;

class Tensor {
public:
    // ── Constructors ──
    Tensor();
    Tensor(const Shape& shape, bool requires_grad = false);
    Tensor(const std::vector<float>& data, const Shape& shape, bool requires_grad = false);

    // ── Static Factories ──
    static Tensor zeros(const Shape& shape, bool requires_grad = false);
    static Tensor ones(const Shape& shape, bool requires_grad = false);
    static Tensor full(const Shape& shape, float value, bool requires_grad = false);
    static Tensor randn(const Shape& shape, bool requires_grad = false);
    static Tensor rand(const Shape& shape, bool requires_grad = false);
    static Tensor from_data(std::initializer_list<float> data, const Shape& shape, bool requires_grad = false);
    static Tensor from_data(const std::vector<float>& data, const Shape& shape, bool requires_grad = false);
    static Tensor eye(size_t n, bool requires_grad = false);

    // ── Properties ──
    const Shape& shape() const;
    size_t size() const;
    size_t ndim() const;
    size_t shape(size_t dim) const;
    bool requires_grad() const;
    void set_requires_grad(bool val);
    bool is_leaf() const;
    float item() const;
    float* data();
    const float* data() const;
    bool empty() const;

    // ── Autograd ──
    Tensor& grad();
    const Tensor& grad() const;
    bool has_grad() const;
    std::shared_ptr<GradFunction> grad_fn() const;
    void set_grad_fn(std::shared_ptr<GradFunction> fn);
    void backward();
    void backward(const Tensor& grad_output);
    Tensor detach() const;
    Tensor clone() const;

    // ── Indexing ──
    float& at(const std::vector<size_t>& indices);
    const float& at(const std::vector<size_t>& indices) const;
    float& operator()(std::initializer_list<size_t> indices);
    const float& operator()(std::initializer_list<size_t> indices) const;

    // ── Shape Operations ──
    Tensor reshape(const Shape& new_shape) const;
    Tensor transpose() const;
    Tensor transpose(size_t dim0, size_t dim1) const;
    Tensor flatten(size_t start_dim = 0) const;
    Tensor unsqueeze(size_t dim) const;
    Tensor squeeze(size_t dim) const;
    Tensor repeat(const Shape& repeats) const;

    // ── Arithmetic Operators (with autograd) ──
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator-() const;

    Tensor operator+(float scalar) const;
    Tensor operator-(float scalar) const;
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;

    friend Tensor operator+(float scalar, const Tensor& t);
    friend Tensor operator*(float scalar, const Tensor& t);
    friend Tensor operator-(float scalar, const Tensor& t);

    // ── In-place (no autograd) ──
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);
    Tensor& operator+=(float scalar);
    Tensor& operator*=(float scalar);

    // ── Math Functions ──
    Tensor matmul(const Tensor& other) const;
    Tensor exp() const;
    Tensor log() const;
    Tensor pow(float exponent) const;
    Tensor sqrt() const;
    Tensor abs() const;
    Tensor clamp(float min_val, float max_val) const;
    Tensor tanh_() const;

    // ── Reductions ──
    Tensor sum(int axis = -1, bool keepdim = false) const;
    Tensor mean(int axis = -1, bool keepdim = false) const;
    Tensor max(int axis = -1, bool keepdim = false) const;
    Tensor min(int axis = -1, bool keepdim = false) const;
    Tensor argmax(int axis = -1) const;

    // ── Comparison (no grad) ──
    Tensor operator>(float val) const;
    Tensor operator<(float val) const;
    Tensor operator>=(float val) const;
    Tensor operator==(const Tensor& other) const;

    // ── In-place Utilities ──
    void zero_();
    void fill_(float val);
    void uniform_(float low = 0.0f, float high = 1.0f);
    void normal_(float mean = 0.0f, float stddev = 1.0f);

    // ── IO ──
    friend std::ostream& operator<<(std::ostream& os, const Tensor& t);
    std::string to_string() const;

    // ── Internal ──
    struct Impl;
    std::shared_ptr<Impl> impl() const { return impl_; }

private:
    std::shared_ptr<Impl> impl_;
    void ensure_impl() const;
};

// ── Tensor::Impl ──
struct Tensor::Impl {
    std::shared_ptr<std::vector<float>> storage;
    Shape shape;
    std::vector<size_t> strides;
    size_t offset = 0;

    bool requires_grad = false;
    std::shared_ptr<Tensor> grad;
    std::shared_ptr<GradFunction> grad_fn;
    bool is_leaf = true;

    Impl() : storage(std::make_shared<std::vector<float>>()) {}

    Impl(const Shape& s, bool req_grad = false)
        : storage(std::make_shared<std::vector<float>>(shape_size(s), 0.0f)),
          shape(s), strides(compute_strides(s)),
          requires_grad(req_grad) {}

    Impl(const std::vector<float>& data, const Shape& s, bool req_grad = false)
        : storage(std::make_shared<std::vector<float>>(data)),
          shape(s), strides(compute_strides(s)),
          requires_grad(req_grad) {
        if (data.size() != shape_size(s)) {
            throw std::runtime_error("Data size does not match shape");
        }
    }

    size_t flat_index(const std::vector<size_t>& indices) const {
        size_t idx = offset;
        for (size_t i = 0; i < indices.size(); ++i) {
            idx += indices[i] * strides[i];
        }
        return idx;
    }

    float* ptr() { return storage->data() + offset; }
    const float* ptr() const { return storage->data() + offset; }
    size_t total_size() const { return shape_size(shape); }
};

// ── Free functions ──
Tensor matmul(const Tensor& a, const Tensor& b);

} // namespace nf
