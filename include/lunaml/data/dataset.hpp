#pragma once
#include "lunaml/core/tensor.hpp"
#include <utility>

namespace nf { namespace data {

class Dataset {
public:
    virtual ~Dataset() = default;
    virtual size_t size() const = 0;
    virtual std::pair<Tensor, Tensor> get(size_t index) const = 0;
};

}} // namespace nf::data
