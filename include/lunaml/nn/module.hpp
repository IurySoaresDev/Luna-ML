#pragma once
#include "lunaml/core/tensor.hpp"
#include <vector>
#include <string>

namespace nf { namespace nn {

class Module {
public:
    virtual ~Module() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual std::vector<Tensor*> parameters();
    virtual std::string name() const { return "Module"; }

    void train() { training_ = true; }
    void eval() { training_ = false; }
    bool is_training() const { return training_; }
    void zero_grad();

protected:
    bool training_ = true;
    std::vector<Tensor> params_;
};

}} // namespace nf::nn
