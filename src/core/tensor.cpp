#include "lunaml/core/autograd.hpp"
#include "lunaml/core/random.hpp"

namespace nf {

// ─────────────────────────────────────────────
// Tensor Constructors
// ─────────────────────────────────────────────

Tensor::Tensor() : impl_(std::make_shared<Impl>()) {}

Tensor::Tensor(const Shape& shape, bool requires_grad)
    : impl_(std::make_shared<Impl>(shape, requires_grad)) {}

Tensor::Tensor(const std::vector<float>& data, const Shape& shape, bool requires_grad)
    : impl_(std::make_shared<Impl>(data, shape, requires_grad)) {}

void Tensor::ensure_impl() const {
    if (!impl_) throw std::runtime_error("Tensor not initialized");
}

// ─────────────────────────────────────────────
// Static Factories
// ─────────────────────────────────────────────

Tensor Tensor::zeros(const Shape& shape, bool requires_grad) {
    return Tensor(shape, requires_grad);
}

Tensor Tensor::ones(const Shape& shape, bool requires_grad) {
    Tensor t(shape, requires_grad);
    std::fill(t.data(), t.data() + t.size(), 1.0f);
    return t;
}

Tensor Tensor::full(const Shape& shape, float value, bool requires_grad) {
    Tensor t(shape, requires_grad);
    std::fill(t.data(), t.data() + t.size(), value);
    return t;
}

Tensor Tensor::randn(const Shape& shape, bool requires_grad) {
    Tensor t(shape, requires_grad);
    auto& gen = Random::generator();
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < t.size(); ++i) t.data()[i] = dist(gen);
    return t;
}

Tensor Tensor::rand(const Shape& shape, bool requires_grad) {
    Tensor t(shape, requires_grad);
    auto& gen = Random::generator();
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < t.size(); ++i) t.data()[i] = dist(gen);
    return t;
}

Tensor Tensor::from_data(std::initializer_list<float> data, const Shape& shape, bool requires_grad) {
    return Tensor(std::vector<float>(data), shape, requires_grad);
}

Tensor Tensor::from_data(const std::vector<float>& data, const Shape& shape, bool requires_grad) {
    return Tensor(data, shape, requires_grad);
}

Tensor Tensor::eye(size_t n, bool requires_grad) {
    Tensor t({n, n}, requires_grad);
    for (size_t i = 0; i < n; ++i) t.data()[i * n + i] = 1.0f;
    return t;
}

// ─────────────────────────────────────────────
// Properties
// ─────────────────────────────────────────────

const Shape& Tensor::shape() const { ensure_impl(); return impl_->shape; }
size_t Tensor::size() const { ensure_impl(); return impl_->total_size(); }
size_t Tensor::ndim() const { ensure_impl(); return impl_->shape.size(); }
size_t Tensor::shape(size_t dim) const { ensure_impl(); return impl_->shape[dim]; }
bool Tensor::requires_grad() const { ensure_impl(); return impl_->requires_grad; }
void Tensor::set_requires_grad(bool val) { ensure_impl(); impl_->requires_grad = val; }
bool Tensor::is_leaf() const { ensure_impl(); return impl_->is_leaf; }
bool Tensor::empty() const { return !impl_ || impl_->shape.empty(); }

float Tensor::item() const {
    ensure_impl();
    if (size() != 1) throw std::runtime_error("item() requires scalar tensor (size=1), got size=" + std::to_string(size()));
    return impl_->ptr()[0];
}

float* Tensor::data() { ensure_impl(); return impl_->ptr(); }
const float* Tensor::data() const { ensure_impl(); return impl_->ptr(); }

// ─────────────────────────────────────────────
// Autograd Properties
// ─────────────────────────────────────────────

Tensor& Tensor::grad() {
    ensure_impl();
    if (!impl_->grad) impl_->grad = std::make_shared<Tensor>();
    return *impl_->grad;
}

const Tensor& Tensor::grad() const {
    ensure_impl();
    if (!impl_->grad) throw std::runtime_error("No gradient computed");
    return *impl_->grad;
}

bool Tensor::has_grad() const {
    return impl_ && impl_->grad && !impl_->grad->empty();
}

std::shared_ptr<GradFunction> Tensor::grad_fn() const {
    ensure_impl();
    return impl_->grad_fn;
}

void Tensor::set_grad_fn(std::shared_ptr<GradFunction> fn) {
    ensure_impl();
    impl_->grad_fn = fn;
    impl_->is_leaf = false;
}

Tensor Tensor::detach() const {
    Tensor t;
    t.impl_ = std::make_shared<Impl>(
        std::vector<float>(data(), data() + size()), shape(), false
    );
    return t;
}

Tensor Tensor::clone() const {
    Tensor t;
    t.impl_ = std::make_shared<Impl>(
        std::vector<float>(data(), data() + size()), shape(), requires_grad()
    );
    return t;
}

// ─────────────────────────────────────────────
// Backward
// ─────────────────────────────────────────────

void Tensor::backward() {
    backward(Tensor::ones(shape()));
}

void Tensor::backward(const Tensor& grad_output) {
    ensure_impl();
    if (!impl_->requires_grad) return;

    // Topological sort
    std::vector<GradFunction*> order;
    std::unordered_set<GradFunction*> visited;

    std::function<void(GradFunction*)> topo_sort = [&](GradFunction* fn) {
        if (!fn || visited.count(fn)) return;
        visited.insert(fn);
        for (auto& input : fn->inputs) {
            if (input.grad_fn()) {
                topo_sort(input.grad_fn().get());
            }
        }
        order.push_back(fn);
    };

    topo_sort(impl_->grad_fn.get());
    std::reverse(order.begin(), order.end());

    // Gradient map: fn -> accumulated gradient
    std::unordered_map<GradFunction*, Tensor> fn_grads;
    if (impl_->grad_fn) {
        fn_grads[impl_->grad_fn.get()] = grad_output;
    } else {
        // Leaf tensor
        if (!impl_->grad) impl_->grad = std::make_shared<Tensor>(Tensor::zeros(shape()));
        for (size_t i = 0; i < size(); ++i) {
            impl_->grad->data()[i] += grad_output.data()[i];
        }
        return;
    }

    for (auto* fn : order) {
        auto it = fn_grads.find(fn);
        if (it == fn_grads.end()) continue;
        auto& g = it->second;

        auto input_grads = fn->apply(g);

        for (size_t i = 0; i < fn->inputs.size() && i < input_grads.size(); ++i) {
            auto& input = fn->inputs[i];
            if (!input.requires_grad()) continue;

            if (input.grad_fn()) {
                auto* input_fn = input.grad_fn().get();
                if (fn_grads.count(input_fn)) {
                    // Accumulate
                    auto& existing = fn_grads[input_fn];
                    for (size_t j = 0; j < existing.size(); ++j) {
                        existing.data()[j] += input_grads[i].data()[j];
                    }
                } else {
                    fn_grads[input_fn] = input_grads[i];
                }
            } else {
                // Leaf tensor — accumulate into .grad
                auto input_impl = input.impl();
                if (!input_impl->grad) {
                    input_impl->grad = std::make_shared<Tensor>(Tensor::zeros(input.shape()));
                }
                for (size_t j = 0; j < input.size(); ++j) {
                    input_impl->grad->data()[j] += input_grads[i].data()[j];
                }
            }
        }
    }
}

// ─────────────────────────────────────────────
// Indexing
// ─────────────────────────────────────────────

float& Tensor::at(const std::vector<size_t>& indices) {
    ensure_impl();
    return (*impl_->storage)[impl_->flat_index(indices)];
}

const float& Tensor::at(const std::vector<size_t>& indices) const {
    ensure_impl();
    return (*impl_->storage)[impl_->flat_index(indices)];
}

float& Tensor::operator()(std::initializer_list<size_t> indices) {
    return at(std::vector<size_t>(indices));
}

const float& Tensor::operator()(std::initializer_list<size_t> indices) const {
    return at(std::vector<size_t>(indices));
}

// ─────────────────────────────────────────────
// Shape Operations
// ─────────────────────────────────────────────

Tensor Tensor::reshape(const Shape& new_shape) const {
    ensure_impl();
    if (shape_size(new_shape) != size()) {
        throw std::runtime_error("Cannot reshape: incompatible sizes");
    }
    Tensor result(std::vector<float>(data(), data() + size()), new_shape, requires_grad());
    if (requires_grad() && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<ReshapeBackward>();
        fn->original_shape = shape();
        fn->inputs = {*this};
        result.set_grad_fn(fn);
    }
    return result;
}

Tensor Tensor::transpose() const {
    ensure_impl();
    if (ndim() != 2) throw std::runtime_error("transpose() requires 2D tensor");
    size_t rows = shape(0), cols = shape(1);
    Tensor result({cols, rows}, requires_grad());
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            result.data()[j * rows + i] = data()[i * cols + j];

    if (requires_grad() && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<TransposeBackward>();
        fn->inputs = {*this};
        result.set_grad_fn(fn);
    }
    return result;
}

Tensor Tensor::transpose(size_t dim0, size_t dim1) const {
    ensure_impl();
    if (dim0 == dim1) return clone();
    if (dim0 >= ndim() || dim1 >= ndim())
        throw std::runtime_error("transpose dims out of range");

    Shape new_shape = shape();
    std::swap(new_shape[dim0], new_shape[dim1]);

    Tensor result(new_shape, requires_grad());
    auto out_strides = compute_strides(new_shape);
    size_t total = size();

    for (size_t i = 0; i < total; ++i) {
        auto idx = unravel_index(i, shape());
        std::swap(idx[dim0], idx[dim1]);
        size_t out_flat = 0;
        for (size_t d = 0; d < idx.size(); ++d) out_flat += idx[d] * out_strides[d];
        result.data()[out_flat] = data()[i];
    }
    return result;
}

Tensor Tensor::flatten(size_t start_dim) const {
    ensure_impl();
    size_t before = 1, after = 1;
    for (size_t i = 0; i < start_dim && i < ndim(); ++i) before *= shape(i);
    for (size_t i = start_dim; i < ndim(); ++i) after *= shape(i);

    Shape new_shape;
    if (start_dim > 0) {
        for (size_t i = 0; i < start_dim; ++i) new_shape.push_back(shape(i));
        new_shape.push_back(after);
    } else {
        new_shape = {size()};
    }

    Tensor result(std::vector<float>(data(), data() + size()), new_shape, requires_grad());
    if (requires_grad() && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<FlattenBackward>();
        fn->original_shape = shape();
        fn->inputs = {*this};
        result.set_grad_fn(fn);
    }
    return result;
}

Tensor Tensor::unsqueeze(size_t dim) const {
    ensure_impl();
    Shape new_shape = shape();
    new_shape.insert(new_shape.begin() + dim, 1);
    return reshape(new_shape);
}

Tensor Tensor::squeeze(size_t dim) const {
    ensure_impl();
    Shape new_shape = shape();
    if (new_shape[dim] == 1) new_shape.erase(new_shape.begin() + dim);
    return reshape(new_shape);
}

Tensor Tensor::repeat(const Shape& repeats) const {
    ensure_impl();
    if (repeats.size() != ndim())
        throw std::runtime_error("repeat dims must match tensor dims");

    Shape out_shape(ndim());
    for (size_t i = 0; i < ndim(); ++i) out_shape[i] = shape(i) * repeats[i];

    Tensor result(out_shape, requires_grad());
    auto out_strides = compute_strides(out_shape);
    size_t total = shape_size(out_shape);

    for (size_t i = 0; i < total; ++i) {
        auto idx = unravel_index(i, out_shape);
        std::vector<size_t> src_idx(ndim());
        for (size_t d = 0; d < ndim(); ++d) src_idx[d] = idx[d] % shape(d);
        size_t src_flat = 0;
        for (size_t d = 0; d < ndim(); ++d) src_flat += src_idx[d] * impl_->strides[d];
        result.data()[i] = data()[src_flat];
    }

    if (requires_grad() && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<RepeatBackward>();
        fn->original_shape = shape();
        fn->repeats = repeats;
        fn->inputs = {*this};
        result.set_grad_fn(fn);
    }
    return result;
}

// ─────────────────────────────────────────────
// Broadcast helper for binary ops
// ─────────────────────────────────────────────

namespace {

Tensor binary_op_impl(const Tensor& a, const Tensor& b,
                      std::function<float(float, float)> op) {
    Shape out_shape = broadcast_shape(a.shape(), b.shape());
    Tensor result(out_shape, a.requires_grad() || b.requires_grad());
    size_t total = shape_size(out_shape);

    auto a_strides = compute_strides(a.shape());
    auto b_strides = compute_strides(b.shape());
    size_t a_diff = out_shape.size() - a.ndim();
    size_t b_diff = out_shape.size() - b.ndim();

    for (size_t i = 0; i < total; ++i) {
        auto idx = unravel_index(i, out_shape);
        size_t ai = broadcast_flat_index(idx, a.shape(), a_strides, a_diff);
        size_t bi = broadcast_flat_index(idx, b.shape(), b_strides, b_diff);
        result.data()[i] = op(a.data()[ai], b.data()[bi]);
    }
    return result;
}

} // anonymous namespace

// ─────────────────────────────────────────────
// Arithmetic Operators
// ─────────────────────────────────────────────

Tensor Tensor::operator+(const Tensor& other) const {
    auto result = binary_op_impl(*this, other, [](float a, float b) { return a + b; });
    if (result.requires_grad() && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<AddBackward>();
        fn->shape_a = shape();
        fn->shape_b = other.shape();
        fn->inputs = {*this, other};
        result.set_grad_fn(fn);
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    auto result = binary_op_impl(*this, other, [](float a, float b) { return a - b; });
    if (result.requires_grad() && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<SubBackward>();
        fn->shape_a = shape();
        fn->shape_b = other.shape();
        fn->inputs = {*this, other};
        result.set_grad_fn(fn);
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    auto result = binary_op_impl(*this, other, [](float a, float b) { return a * b; });
    if (result.requires_grad() && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<MulBackward>();
        fn->inputs = {*this, other};
        fn->save_for_backward(*this);
        fn->save_for_backward(other);
        result.set_grad_fn(fn);
    }
    return result;
}

Tensor Tensor::operator/(const Tensor& other) const {
    auto result = binary_op_impl(*this, other, [](float a, float b) { return a / b; });
    if (result.requires_grad() && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<DivBackward>();
        fn->inputs = {*this, other};
        fn->save_for_backward(*this);
        fn->save_for_backward(other);
        result.set_grad_fn(fn);
    }
    return result;
}

Tensor Tensor::operator-() const {
    Tensor result(shape(), requires_grad());
    for (size_t i = 0; i < size(); ++i) result.data()[i] = -data()[i];
    if (requires_grad() && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<NegBackward>();
        fn->inputs = {*this};
        result.set_grad_fn(fn);
    }
    return result;
}

Tensor Tensor::operator+(float scalar) const {
    Tensor result(shape(), requires_grad());
    for (size_t i = 0; i < size(); ++i) result.data()[i] = data()[i] + scalar;
    if (requires_grad() && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<ScalarAddBackward>();
        fn->inputs = {*this};
        result.set_grad_fn(fn);
    }
    return result;
}

Tensor Tensor::operator-(float scalar) const { return *this + (-scalar); }

Tensor Tensor::operator*(float scalar) const {
    Tensor result(shape(), requires_grad());
    for (size_t i = 0; i < size(); ++i) result.data()[i] = data()[i] * scalar;
    if (requires_grad() && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<ScalarMulBackward>();
        fn->scalar = scalar;
        fn->inputs = {*this};
        result.set_grad_fn(fn);
    }
    return result;
}

Tensor Tensor::operator/(float scalar) const { return *this * (1.0f / scalar); }

Tensor operator+(float scalar, const Tensor& t) { return t + scalar; }
Tensor operator*(float scalar, const Tensor& t) { return t * scalar; }
Tensor operator-(float scalar, const Tensor& t) { return (-t) + scalar; }

// In-place ops (no autograd)
Tensor& Tensor::operator+=(const Tensor& other) {
    for (size_t i = 0; i < size(); ++i) data()[i] += other.data()[i];
    return *this;
}
Tensor& Tensor::operator-=(const Tensor& other) {
    for (size_t i = 0; i < size(); ++i) data()[i] -= other.data()[i];
    return *this;
}
Tensor& Tensor::operator*=(const Tensor& other) {
    for (size_t i = 0; i < size(); ++i) data()[i] *= other.data()[i];
    return *this;
}
Tensor& Tensor::operator/=(const Tensor& other) {
    for (size_t i = 0; i < size(); ++i) data()[i] /= other.data()[i];
    return *this;
}
Tensor& Tensor::operator+=(float scalar) {
    for (size_t i = 0; i < size(); ++i) data()[i] += scalar;
    return *this;
}
Tensor& Tensor::operator*=(float scalar) {
    for (size_t i = 0; i < size(); ++i) data()[i] *= scalar;
    return *this;
}

// ─────────────────────────────────────────────
// Matrix Multiplication
// ─────────────────────────────────────────────

Tensor Tensor::matmul(const Tensor& other) const {
    ensure_impl();
    if (ndim() < 2 || other.ndim() < 2) {
        // Handle vector-matrix cases
        if (ndim() == 1 && other.ndim() == 2) {
            auto a2d = reshape({1, shape(0)});
            return a2d.matmul(other).reshape({other.shape(1)});
        }
        if (ndim() == 2 && other.ndim() == 1) {
            auto b2d = other.reshape({other.shape(0), 1});
            return matmul(b2d).reshape({shape(0)});
        }
        throw std::runtime_error("matmul requires at least 1D tensors");
    }

    size_t M = shape(ndim() - 2);
    size_t K = shape(ndim() - 1);
    size_t N = other.shape(other.ndim() - 1);

    if (K != other.shape(other.ndim() - 2)) {
        throw std::runtime_error("matmul shape mismatch: " +
            std::to_string(K) + " vs " + std::to_string(other.shape(other.ndim() - 2)));
    }

    // For 2D matmul
    Tensor result({M, N}, requires_grad() || other.requires_grad());
    const float* A = data();
    const float* B = other.data();
    float* C = result.data();

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    if (result.requires_grad() && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<MatMulBackward>();
        fn->inputs = {*this, other};
        fn->save_for_backward(*this);
        fn->save_for_backward(other);
        result.set_grad_fn(fn);
    }
    return result;
}

Tensor matmul(const Tensor& a, const Tensor& b) { return a.matmul(b); }

// ─────────────────────────────────────────────
// Math Functions
// ─────────────────────────────────────────────

Tensor Tensor::exp() const {
    Tensor result(shape(), requires_grad());
    for (size_t i = 0; i < size(); ++i) result.data()[i] = std::exp(data()[i]);
    if (requires_grad() && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<ExpBackward>();
        fn->output = result;
        fn->inputs = {*this};
        result.set_grad_fn(fn);
    }
    return result;
}

Tensor Tensor::log() const {
    Tensor result(shape(), requires_grad());
    for (size_t i = 0; i < size(); ++i) result.data()[i] = std::log(data()[i] + 1e-8f);
    if (requires_grad() && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<LogBackward>();
        fn->inputs = {*this};
        fn->save_for_backward(*this);
        result.set_grad_fn(fn);
    }
    return result;
}

Tensor Tensor::pow(float exponent) const {
    Tensor result(shape(), requires_grad());
    for (size_t i = 0; i < size(); ++i) result.data()[i] = std::pow(data()[i], exponent);
    if (requires_grad() && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<PowBackward>();
        fn->exponent = exponent;
        fn->inputs = {*this};
        fn->save_for_backward(*this);
        result.set_grad_fn(fn);
    }
    return result;
}

Tensor Tensor::sqrt() const { return pow(0.5f); }

Tensor Tensor::abs() const {
    Tensor result(shape(), false);
    for (size_t i = 0; i < size(); ++i) result.data()[i] = std::abs(data()[i]);
    return result;
}

Tensor Tensor::clamp(float min_val, float max_val) const {
    Tensor result(shape(), requires_grad());
    for (size_t i = 0; i < size(); ++i)
        result.data()[i] = std::max(min_val, std::min(max_val, data()[i]));
    if (requires_grad() && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<ClampBackward>();
        fn->min_val = min_val;
        fn->max_val = max_val;
        fn->inputs = {*this};
        fn->save_for_backward(*this);
        result.set_grad_fn(fn);
    }
    return result;
}

Tensor Tensor::tanh_() const {
    Tensor result(shape(), requires_grad());
    for (size_t i = 0; i < size(); ++i) result.data()[i] = std::tanh(data()[i]);
    if (requires_grad() && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<TanhBackward>();
        fn->output = result;
        fn->inputs = {*this};
        result.set_grad_fn(fn);
    }
    return result;
}

// ─────────────────────────────────────────────
// Reductions
// ─────────────────────────────────────────────

Tensor Tensor::sum(int axis, bool keepdim) const {
    ensure_impl();
    bool req = requires_grad() && !NoGradGuard::is_enabled();

    if (axis == -1) {
        float total = 0;
        for (size_t i = 0; i < size(); ++i) total += data()[i];
        Tensor result({1}, req);
        result.data()[0] = total;
        if (req) {
            auto fn = std::make_shared<SumBackward>();
            fn->input_shape = shape();
            fn->axis = -1;
            fn->keepdim = keepdim;
            fn->inputs = {*this};
            result.set_grad_fn(fn);
        }
        return result;
    }

    size_t a = static_cast<size_t>(axis);
    if (a >= ndim()) throw std::runtime_error("sum axis out of range");

    Shape out_shape;
    for (size_t i = 0; i < ndim(); ++i) {
        if (i == a) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(shape(i));
    }
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor result(out_shape, req);

    size_t outer = 1, inner = 1;
    for (size_t i = 0; i < a; ++i) outer *= shape(i);
    for (size_t i = a + 1; i < ndim(); ++i) inner *= shape(i);
    size_t dim_size = shape(a);

    for (size_t o = 0; o < outer; ++o) {
        for (size_t in = 0; in < inner; ++in) {
            float s = 0;
            for (size_t d = 0; d < dim_size; ++d) {
                s += data()[(o * dim_size + d) * inner + in];
            }
            result.data()[o * inner + in] = s;
        }
    }

    if (req) {
        auto fn = std::make_shared<SumBackward>();
        fn->input_shape = shape();
        fn->axis = axis;
        fn->keepdim = keepdim;
        fn->inputs = {*this};
        result.set_grad_fn(fn);
    }
    return result;
}

Tensor Tensor::mean(int axis, bool keepdim) const {
    ensure_impl();
    bool req = requires_grad() && !NoGradGuard::is_enabled();

    if (axis == -1) {
        float total = 0;
        for (size_t i = 0; i < size(); ++i) total += data()[i];
        Tensor result({1}, req);
        result.data()[0] = total / static_cast<float>(size());
        if (req) {
            auto fn = std::make_shared<MeanBackward>();
            fn->input_shape = shape();
            fn->axis = -1;
            fn->keepdim = keepdim;
            fn->inputs = {*this};
            result.set_grad_fn(fn);
        }
        return result;
    }

    size_t a = static_cast<size_t>(axis);
    Tensor s = sum(axis, keepdim);

    float n = static_cast<float>(shape(a));
    Tensor result(s.shape(), req);
    for (size_t i = 0; i < s.size(); ++i)
        result.data()[i] = s.data()[i] / n;

    if (req) {
        auto fn = std::make_shared<MeanBackward>();
        fn->input_shape = shape();
        fn->axis = axis;
        fn->keepdim = keepdim;
        fn->inputs = {*this};
        result.set_grad_fn(fn);
    }
    return result;
}

Tensor Tensor::max(int axis, bool keepdim) const {
    ensure_impl();
    if (axis == -1) {
        float m = data()[0];
        for (size_t i = 1; i < size(); ++i) m = std::max(m, data()[i]);
        Tensor result({1}, false);
        result.data()[0] = m;
        return result;
    }
    size_t a = static_cast<size_t>(axis);
    Shape out_shape;
    for (size_t i = 0; i < ndim(); ++i) {
        if (i == a) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(shape(i));
    }
    if (out_shape.empty()) out_shape.push_back(1);
    Tensor result(out_shape, false);
    size_t outer = 1, inner = 1;
    for (size_t i = 0; i < a; ++i) outer *= shape(i);
    for (size_t i = a + 1; i < ndim(); ++i) inner *= shape(i);
    size_t dim_size = shape(a);
    for (size_t o = 0; o < outer; ++o) {
        for (size_t in = 0; in < inner; ++in) {
            float m = data()[(o * dim_size) * inner + in];
            for (size_t d = 1; d < dim_size; ++d)
                m = std::max(m, data()[(o * dim_size + d) * inner + in]);
            result.data()[o * inner + in] = m;
        }
    }
    return result;
}

Tensor Tensor::min(int axis, bool keepdim) const {
    ensure_impl();
    if (axis == -1) {
        float m = data()[0];
        for (size_t i = 1; i < size(); ++i) m = std::min(m, data()[i]);
        Tensor result({1}, false);
        result.data()[0] = m;
        return result;
    }
    size_t a = static_cast<size_t>(axis);
    Shape out_shape;
    for (size_t i = 0; i < ndim(); ++i) {
        if (i == a) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(shape(i));
    }
    if (out_shape.empty()) out_shape.push_back(1);
    Tensor result(out_shape, false);
    size_t outer = 1, inner = 1;
    for (size_t i = 0; i < a; ++i) outer *= shape(i);
    for (size_t i = a + 1; i < ndim(); ++i) inner *= shape(i);
    size_t dim_size = shape(a);
    for (size_t o = 0; o < outer; ++o) {
        for (size_t in = 0; in < inner; ++in) {
            float m = data()[(o * dim_size) * inner + in];
            for (size_t d = 1; d < dim_size; ++d)
                m = std::min(m, data()[(o * dim_size + d) * inner + in]);
            result.data()[o * inner + in] = m;
        }
    }
    return result;
}

Tensor Tensor::argmax(int axis) const {
    ensure_impl();
    if (axis == -1) {
        float m = data()[0];
        size_t mi = 0;
        for (size_t i = 1; i < size(); ++i) {
            if (data()[i] > m) { m = data()[i]; mi = i; }
        }
        Tensor result({1}, false);
        result.data()[0] = static_cast<float>(mi);
        return result;
    }
    size_t a = static_cast<size_t>(axis);
    Shape out_shape;
    for (size_t i = 0; i < ndim(); ++i) {
        if (i != a) out_shape.push_back(shape(i));
    }
    if (out_shape.empty()) out_shape.push_back(1);
    Tensor result(out_shape, false);
    size_t outer = 1, inner = 1;
    for (size_t i = 0; i < a; ++i) outer *= shape(i);
    for (size_t i = a + 1; i < ndim(); ++i) inner *= shape(i);
    size_t dim_size = shape(a);
    for (size_t o = 0; o < outer; ++o) {
        for (size_t in = 0; in < inner; ++in) {
            float m = data()[(o * dim_size) * inner + in];
            size_t mi = 0;
            for (size_t d = 1; d < dim_size; ++d) {
                float v = data()[(o * dim_size + d) * inner + in];
                if (v > m) { m = v; mi = d; }
            }
            result.data()[o * inner + in] = static_cast<float>(mi);
        }
    }
    return result;
}

// ─────────────────────────────────────────────
// Comparison
// ─────────────────────────────────────────────

Tensor Tensor::operator>(float val) const {
    Tensor result(shape(), false);
    for (size_t i = 0; i < size(); ++i) result.data()[i] = data()[i] > val ? 1.0f : 0.0f;
    return result;
}

Tensor Tensor::operator<(float val) const {
    Tensor result(shape(), false);
    for (size_t i = 0; i < size(); ++i) result.data()[i] = data()[i] < val ? 1.0f : 0.0f;
    return result;
}

Tensor Tensor::operator>=(float val) const {
    Tensor result(shape(), false);
    for (size_t i = 0; i < size(); ++i) result.data()[i] = data()[i] >= val ? 1.0f : 0.0f;
    return result;
}

Tensor Tensor::operator==(const Tensor& other) const {
    Tensor result(shape(), false);
    for (size_t i = 0; i < size(); ++i) result.data()[i] = (data()[i] == other.data()[i]) ? 1.0f : 0.0f;
    return result;
}

// ─────────────────────────────────────────────
// In-place Utilities
// ─────────────────────────────────────────────

void Tensor::zero_() { std::fill(data(), data() + size(), 0.0f); }
void Tensor::fill_(float val) { std::fill(data(), data() + size(), val); }

void Tensor::uniform_(float low, float high) {
    auto& gen = Random::generator();
    std::uniform_real_distribution<float> dist(low, high);
    for (size_t i = 0; i < size(); ++i) data()[i] = dist(gen);
}

void Tensor::normal_(float mean, float stddev) {
    auto& gen = Random::generator();
    std::normal_distribution<float> dist(mean, stddev);
    for (size_t i = 0; i < size(); ++i) data()[i] = dist(gen);
}

// ─────────────────────────────────────────────
// IO
// ─────────────────────────────────────────────

std::string Tensor::to_string() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "Tensor(shape=[";
    for (size_t i = 0; i < ndim(); ++i) {
        if (i > 0) ss << ", ";
        ss << shape(i);
    }
    ss << "], data=[";
    size_t max_show = std::min(size(), size_t(20));
    for (size_t i = 0; i < max_show; ++i) {
        if (i > 0) ss << ", ";
        ss << data()[i];
    }
    if (size() > max_show) ss << ", ...";
    ss << "])";
    return ss.str();
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << t.to_string();
    return os;
}

} // namespace nf
