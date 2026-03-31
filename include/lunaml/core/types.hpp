#pragma once

#include <vector>
#include <cstddef>
#include <string>
#include <memory>
#include <functional>
#include <iostream>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <initializer_list>
#include <iomanip>

namespace nf {

using float32 = float;
using Shape = std::vector<size_t>;

enum class Device {
    CPU,
    CUDA // reserved for future
};

// Compute total number of elements from shape
inline size_t shape_size(const Shape& shape) {
    if (shape.empty()) return 0;
    size_t s = 1;
    for (auto d : shape) s *= d;
    return s;
}

// Compute strides from shape (row-major / C-contiguous)
inline std::vector<size_t> compute_strides(const Shape& shape) {
    std::vector<size_t> strides(shape.size());
    if (shape.empty()) return strides;
    strides.back() = 1;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

// Compute broadcast output shape
inline Shape broadcast_shape(const Shape& a, const Shape& b) {
    size_t max_dim = std::max(a.size(), b.size());
    Shape result(max_dim);
    for (size_t i = 0; i < max_dim; ++i) {
        size_t da = (i < max_dim - a.size()) ? 1 : a[i - (max_dim - a.size())];
        size_t db = (i < max_dim - b.size()) ? 1 : b[i - (max_dim - b.size())];
        if (da != db && da != 1 && db != 1) {
            throw std::runtime_error("Cannot broadcast shapes");
        }
        result[i] = std::max(da, db);
    }
    return result;
}

// Convert flat index to multi-dimensional index
inline std::vector<size_t> unravel_index(size_t flat, const Shape& shape) {
    std::vector<size_t> idx(shape.size());
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        idx[i] = flat % shape[i];
        flat /= shape[i];
    }
    return idx;
}

// Convert multi-dimensional index to flat index with broadcasting
inline size_t broadcast_flat_index(const std::vector<size_t>& out_idx,
                                    const Shape& in_shape,
                                    const std::vector<size_t>& in_strides,
                                    size_t ndim_diff) {
    size_t idx = 0;
    for (size_t i = 0; i < in_shape.size(); ++i) {
        size_t coord = (in_shape[i] == 1) ? 0 : out_idx[i + ndim_diff];
        idx += coord * in_strides[i];
    }
    return idx;
}

} // namespace nf
