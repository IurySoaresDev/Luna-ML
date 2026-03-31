#pragma once
#include "lunaml/core/tensor.hpp"

namespace nf { namespace loss {

class Loss {
public:
    virtual ~Loss() = default;
    virtual Tensor forward(const Tensor& prediction, const Tensor& target) = 0;
    Tensor operator()(const Tensor& prediction, const Tensor& target) {
        return forward(prediction, target);
    }
};

}} // namespace nf::loss
