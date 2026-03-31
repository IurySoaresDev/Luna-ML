#pragma once
#include "lunaml/loss/loss.hpp"

namespace nf { namespace loss {

class BCE : public Loss {
public:
    Tensor forward(const Tensor& prediction, const Tensor& target) override;
};

}} // namespace nf::loss
