#pragma once
#include "lunaml/loss/loss.hpp"
#include "lunaml/core/tensor_ops.hpp"

namespace nf { namespace loss {

class CrossEntropy : public Loss {
public:
    Tensor forward(const Tensor& prediction, const Tensor& target) override;
};

}} // namespace nf::loss
