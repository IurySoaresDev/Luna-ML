#pragma once
#include "lunaml/optim/optimizer.hpp"

namespace nf { namespace optim {

class SGD : public Optimizer {
public:
    SGD(std::vector<Tensor*> params, float lr = 0.01f, float momentum = 0.0f, float weight_decay = 0.0f);
    void step() override;

private:
    float momentum_, weight_decay_;
    std::vector<Tensor> velocity_;
};

}} // namespace nf::optim
